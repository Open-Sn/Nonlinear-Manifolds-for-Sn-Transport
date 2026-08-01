"""Provenance-aware result-directory helpers for explicit 1-D executions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import scipy

from .config import OneDConfig


@dataclass(frozen=True)
class RunDirectory:
    root: Path
    config_path: Path
    manifest_path: Path
    logs: Path
    data: Path
    metrics: Path
    figures: Path


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _git_metadata(repository_root: str | Path | None = None) -> dict[str, Any]:
    cwd = Path(repository_root) if repository_root is not None else Path.cwd()
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize provenance value {type(value).__name__}")


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            content,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def create_run_directory(
    config: OneDConfig,
    *,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    run_directory: str | Path | None = None,
    config_source: str | Path | None = None,
    execution_stage: str = "initialized",
    parent_provenance: dict[str, Any] | None = None,
    repository_root: str | Path | None = None,
) -> RunDirectory:
    """Create one run contract only for an explicitly requested execution."""
    created = _utc_now()
    if run_directory is None:
        if run_id is None:
            run_id = created.strftime("%Y%m%dT%H%M%SZ") + "-" + config.checksum()[:8]
        root = Path(output_root or config.output.output_root) / run_id
    else:
        root = Path(run_directory)
        run_id = run_id or root.name
    if root.exists():
        raise FileExistsError(f"run directory already exists: {root}")

    logs = root / "logs"
    data = root / "data"
    metrics = root / "metrics"
    figures = root / "figures"
    for path in (logs, data, metrics, figures):
        path.mkdir(parents=True, exist_ok=False)

    config_path = root / "config.json"
    manifest_path = root / "manifest.json"
    _write_json(config_path, config.to_dict())
    git = _git_metadata(repository_root)
    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "creation_date": created.date().isoformat(),
        "creation_time_utc": created.isoformat(),
        "git": git,
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
        },
        "configuration_file": str(config_source) if config_source else None,
        "configuration": config.to_dict(),
        "configuration_canonical_json": config.canonical_json(),
        "configuration_checksum_sha256": config.checksum(),
        "snapshot": {
            "filename": config.output.snapshot_filename,
            "shape": None,
            "dtype": None,
            "content_sha256": None,
        },
        "execution": {
            "stage": execution_stage,
            "solver_success": None,
            "start_time_utc": None,
            "finish_time_utc": None,
            "elapsed_seconds": None,
            "diagnostics": {},
        },
        "parent_provenance": parent_provenance,
    }
    _write_json(manifest_path, manifest)
    return RunDirectory(
        root=root,
        config_path=config_path,
        manifest_path=manifest_path,
        logs=logs,
        data=data,
        metrics=metrics,
        figures=figures,
    )


def load_manifest(run: RunDirectory | str | Path) -> dict[str, Any]:
    path = run.manifest_path if isinstance(run, RunDirectory) else Path(run) / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(
    run: RunDirectory | str | Path,
    *,
    snapshot: dict[str, Any] | None = None,
    execution: dict[str, Any] | None = None,
    parent_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Update selected manifest sections without discarding recorded provenance."""
    if isinstance(run, RunDirectory):
        path = run.manifest_path
    else:
        path = Path(run) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if snapshot:
        manifest["snapshot"].update(snapshot)
    if execution:
        manifest["execution"].update(execution)
    if parent_provenance is not None:
        manifest["parent_provenance"] = parent_provenance
    _write_json(path, manifest)
    return manifest


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
