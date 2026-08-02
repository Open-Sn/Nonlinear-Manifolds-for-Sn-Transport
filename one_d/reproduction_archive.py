"""Portable, deterministic packaging for completed one-dimensional results.

This module reads and validates existing artifacts only.  It deliberately has
no imports from solver, inference, POD, derivative, metric, bundle-building, or
plotting entry points.
"""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from typing import Any, Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC_PATH = REPOSITORY_ROOT / "configs/1d/publication/archive_spec.json"
CONTROL_FILES = {
    "inventory.json",
    "inventory.tsv",
    "SHA256SUMS",
    "archive_metadata.json",
}
TEXT_SUFFIXES = {".json", ".md", ".txt", ".tsv", ".csv"}
ARCHIVE_ROOTS = {"core": "1d_reproduction", "audit": "1d_audit_supplement"}


@dataclass(frozen=True)
class ArchiveEntry:
    source: Path
    source_relative: str
    archive_path: str
    role: str
    run_id: str | None
    tracked_status: str
    authority: str


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json_checksum(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(payload)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def load_archive_spec(path: str | Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    spec = _load_json(Path(path))
    if spec.get("schema_version") != "1.0.0":
        raise ValueError("unsupported reproduction archive specification")
    for key in ("source", "runs", "core_groups", "audit_groups"):
        if key not in spec:
            raise ValueError(f"archive specification is missing {key}")
    return spec


def is_safe_relative_path(value: str | PurePosixPath) -> bool:
    path = PurePosixPath(str(value))
    return (
        bool(str(path))
        and not path.is_absolute()
        and ".." not in path.parts
        and path.parts[0] not in {"", "."}
        and not re.match(r"^[A-Za-z]:", str(path))
    )


def _format(value: str, runs: Mapping[str, str]) -> str:
    try:
        return value.format(**runs)
    except KeyError as error:
        raise ValueError(f"unknown run placeholder in {value}: {error.args[0]}") from None


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    ).stdout.strip()


def source_state(root: str | Path = REPOSITORY_ROOT) -> dict[str, Any]:
    repository = Path(root)
    head = _git_output(repository, "rev-parse", "HEAD")
    branch = _git_output(repository, "branch", "--show-current")
    staged = _git_output(repository, "diff", "--cached", "--name-status")
    tracked_changes = _git_output(repository, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": head,
        "short_commit": _git_output(repository, "rev-parse", "--short", "HEAD"),
        "branch": branch,
        "index_clean": not bool(staged),
        "tracked_worktree_clean": not bool(tracked_changes),
    }


def _resolved_runs(
    spec: Mapping[str, Any],
    overrides: Mapping[str, str] | None,
    *,
    allow_unknown_run_ids: bool,
) -> dict[str, str]:
    runs = {str(key): str(value) for key, value in spec["runs"].items()}
    for key, value in (overrides or {}).items():
        if key not in runs:
            raise ValueError(f"unknown archive run key: {key}")
        if value != runs[key] and not allow_unknown_run_ids:
            raise ValueError(
                f"refusing unknown authoritative run ID for {key}: {value}; "
                "use --allow-unknown-run-id to override explicitly"
            )
        runs[key] = value
    return runs


def _tracked_files(root: Path) -> set[str]:
    return set(_git_output(root, "ls-files", "-z").split("\0"))


def _excluded(path: Path, exclusions: Iterable[str]) -> bool:
    names = set(path.parts)
    if names.intersection(exclusions):
        return True
    return path.suffix in {".pyc", ".pyo"} or path.name.endswith("~")


def _matches(relative: str, patterns: Iterable[str] | None) -> bool:
    values = list(patterns or [])
    return not values or any(fnmatch.fnmatch(relative, pattern) for pattern in values)


def collect_archive_entries(
    spec: Mapping[str, Any],
    kind: str,
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    run_overrides: Mapping[str, str] | None = None,
    allow_unknown_run_ids: bool = False,
) -> list[ArchiveEntry]:
    if kind not in ARCHIVE_ROOTS:
        raise ValueError(f"unsupported archive kind: {kind}")
    root = Path(repository_root).resolve()
    runs = _resolved_runs(
        spec,
        run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    tracked = _tracked_files(root)
    exclusions = set(spec.get("global_exclusions", []))
    entries: list[ArchiveEntry] = []
    archive_paths: set[str] = set()
    source_paths: set[str] = set()
    for group in spec[f"{kind}_groups"]:
        source_relative = _format(group["source"], runs)
        if not is_safe_relative_path(source_relative):
            raise ValueError(f"unsafe source path in archive specification: {source_relative}")
        source = (root / source_relative).resolve()
        try:
            source.relative_to(root)
        except ValueError:
            raise ValueError(f"archive source escapes repository: {source_relative}") from None
        if not source.exists():
            raise FileNotFoundError(f"required archive source is missing: {source_relative}")
        destination = _format(group["destination"], runs)
        if not is_safe_relative_path(destination):
            raise ValueError(f"unsafe archive destination: {destination}")
        run_id = _format(group.get("run_id", ""), runs) or None
        candidates: list[tuple[Path, str]]
        if source.is_file():
            candidates = [(source, "")]
        elif source.is_dir():
            candidates = [
                (path, path.relative_to(source).as_posix())
                for path in source.rglob("*")
                if path.is_file() or path.is_symlink()
            ]
        else:
            raise ValueError(f"archive source is neither file nor directory: {source_relative}")
        for path, relative in sorted(candidates, key=lambda item: item[1]):
            if relative and not _matches(relative, group.get("include")):
                continue
            if relative and _matches(relative, group.get("exclude")) and group.get("exclude"):
                continue
            relative_path = Path(relative) if relative else Path(path.name)
            if _excluded(relative_path, exclusions):
                continue
            if path.is_symlink():
                resolved = path.resolve()
                try:
                    resolved.relative_to(root)
                except ValueError:
                    raise ValueError(f"archive source symlink escapes repository: {path}") from None
                path = resolved
            original = path.relative_to(root).as_posix()
            archive_path = (
                PurePosixPath(destination, relative).as_posix()
                if relative
                else PurePosixPath(destination).as_posix()
            )
            if not is_safe_relative_path(archive_path):
                raise ValueError(f"unsafe resolved archive path: {archive_path}")
            if archive_path in archive_paths:
                raise ValueError(f"duplicate archive path: {archive_path}")
            if original in source_paths:
                raise ValueError(f"duplicate source file within {kind} archive: {original}")
            archive_paths.add(archive_path)
            source_paths.add(original)
            entries.append(
                ArchiveEntry(
                    source=path,
                    source_relative=original,
                    archive_path=archive_path,
                    role=group["role"],
                    run_id=run_id,
                    tracked_status="tracked" if original in tracked else "generated",
                    authority="authoritative" if kind == "core" else "supplemental",
                )
            )
    return sorted(entries, key=lambda item: item.archive_path)


def assert_core_audit_non_overlap(
    core: Iterable[ArchiveEntry], audit: Iterable[ArchiveEntry]
) -> None:
    core_sources = {entry.source_relative for entry in core}
    audit_sources = {entry.source_relative for entry in audit}
    overlap = sorted(core_sources.intersection(audit_sources))
    if overlap:
        raise ValueError("core and audit source files overlap: " + ", ".join(overlap[:5]))


def _portable_bytes(path: Path, repository_root: Path) -> tuple[bytes, bool]:
    raw = path.read_bytes()
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return raw, False
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw, False
    portable = text.replace(repository_root.as_posix() + "/", "")
    portable = portable.replace(repository_root.as_posix(), ".")
    portable = portable.replace("/opt/anaconda3/bin/python", "python")
    portable = portable.replace("/opt/anaconda3", "<python-environment>")
    portable = re.sub(r"/var/folders/[^\"'\s]+", "<temporary-path>", portable)
    if re.search(r"/(Users|home)/", portable):
        raise ValueError(f"portable archive text still contains workstation path: {path}")
    return portable.encode("utf-8"), portable != text


def _validate_file_checksum(root: Path, path_template: str, runs: Mapping[str, str], expected: str) -> Path:
    relative = _format(path_template, runs)
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"authoritative file is missing: {relative}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"authoritative checksum mismatch for {relative}: {actual}")
    return path


def validate_authoritative_inputs(
    spec: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    run_overrides: Mapping[str, str] | None = None,
    allow_unknown_run_ids: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    runs = _resolved_runs(
        spec,
        run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    state = source_state(root)
    expected_source = spec["source"]
    if state["commit"] != expected_source["commit"]:
        raise ValueError(
            f"source commit mismatch: expected {expected_source['commit']}, found {state['commit']}"
        )
    if state["branch"] != expected_source["branch"]:
        raise ValueError("source branch does not match archive specification")
    if not state["index_clean"]:
        raise ValueError("Git index must remain untouched before archive creation")

    validated_files: dict[str, Any] = {}
    for record in spec.get("important_files", []):
        path = _validate_file_checksum(root, record["path"], runs, record["sha256"])
        validated_files[record["name"]] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": record["sha256"],
            "size_bytes": path.stat().st_size,
        }

    # Synthetic archive-format tests exercise the packaging machinery without
    # requiring or copying production scientific data.  Production is the
    # default and the checked-in archive specification does not set this key.
    if spec.get("validation_profile", "production") == "synthetic":
        return {
            "source": state,
            "validated_files": validated_files,
            "shared_arrays": {},
            "bundles": {},
            "final_figures": {},
            "scientific_checksums": dict(spec.get("scientific_checksums", {})),
        }

    scientific = spec.get("scientific_checksums", {})
    config = _load_json(root / "configs/1d/legacy_production.json")
    if canonical_json_checksum(config) != scientific.get("configuration_canonical_sha256"):
        raise ValueError("production configuration canonical checksum mismatch")
    catalog = _load_json(root / "configs/1d/publication/experiments.json")
    if canonical_json_checksum(catalog) != scientific.get("catalog_canonical_sha256"):
        raise ValueError("publication catalog canonical checksum mismatch")
    selected = _load_json(root / "configs/1d/publication/figure4_selected_parameters.json")
    selected_payload = dict(selected)
    selected_checksum = selected_payload.pop("content_checksum_sha256", None)
    if (
        selected_checksum != canonical_json_checksum(selected_payload)
        or selected_checksum
        != scientific.get("figure4_selected_parameters_content_sha256")
    ):
        raise ValueError("tracked Figure 4 selected-parameter content checksum mismatch")
    golden = _load_json(root / "tests/golden/tiny_1d_manifest.json")
    if golden.get("content_checksum", {}).get("sha256") != scientific.get(
        "independent_golden_content_sha256"
    ):
        raise ValueError("independent golden content checksum mismatch")

    fom_manifest_path = root / _format(
        "results/1d/publication/base_fom/{base_fom}/manifest.json", runs
    )
    fom_manifest = _load_json(fom_manifest_path)
    if fom_manifest.get("snapshot", {}).get("content_sha256") != scientific.get(
        "dataset_sha256"
    ):
        raise ValueError("FOM manifest dataset checksum mismatch")
    if fom_manifest.get("configuration_checksum_sha256") != scientific.get(
        "configuration_canonical_sha256"
    ):
        raise ValueError("FOM manifest configuration checksum mismatch")

    shared_root = root / _format(
        "results/1d/publication/shared_offline/"
        "a3885dc5a071f67afb514e3d130d15cd993737a174313084f7e1ed0911cef6b3/"
        "{shared_offline}",
        runs,
    )
    shared_manifest = _load_json(shared_root / "manifest.json")
    if shared_manifest.get("dataset", {}).get("sha256") != scientific.get(
        "dataset_sha256"
    ):
        raise ValueError("shared-offline dataset checksum mismatch")
    shared_arrays: dict[str, Any] = {}
    for name, record in sorted(shared_manifest.get("arrays", {}).items()):
        path = shared_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"shared-offline array is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"] or path.stat().st_size != record["file_size_bytes"]:
            raise ValueError(f"shared-offline array validation failed: {name}")
        shared_arrays[name] = {
            "path": path.relative_to(root).as_posix(),
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "shape": record["shape"],
            "dtype": record["dtype"],
        }

    bundle_status: dict[str, Any] = {}
    for figure in range(1, 6):
        bundle_key = f"figure{figure}_bundle"
        run_id = runs[bundle_key]
        if figure <= 3:
            directory = root / f"results/1d/publication/figure_data/figure{figure}/{run_id}"
            metadata_path = directory / "figure_data.json"
            metadata = _load_json(metadata_path)
            complete = metadata.get("case_set_complete") is True and metadata.get(
                "status"
            ) == "complete_input_set"
        else:
            directory = root / f"results/1d/publication/figure_data/figure{figure}/{run_id}"
            metadata_path = directory / f"figure{figure}_data.json"
            metadata = _load_json(metadata_path)
            complete = metadata.get("case_set_status") == "complete"
        if metadata.get("figure") != f"Figure {figure}" or not complete:
            raise ValueError(f"authoritative Figure {figure} bundle is incomplete")
        bundle_status[f"figure{figure}"] = {
            "run_id": run_id,
            "metadata": metadata_path.relative_to(root).as_posix(),
            "complete": True,
        }

    final_figures: dict[str, Any] = {}
    for figure, record in sorted(spec.get("final_figure_checksums", {}).items()):
        directory = root / _format(record["directory"], runs)
        files: dict[str, str] = {}
        for filename, expected in sorted(record["files"].items()):
            path = directory / filename
            actual = sha256_file(path)
            if actual != expected:
                raise ValueError(f"final {figure} checksum mismatch: {filename}")
            files[filename] = actual
        final_figures[figure] = {
            "directory": directory.relative_to(root).as_posix(),
            "files": files,
        }
    return {
        "source": state,
        "validated_files": validated_files,
        "shared_arrays": shared_arrays,
        "bundles": bundle_status,
        "final_figures": final_figures,
        "scientific_checksums": dict(scientific),
    }


def _environment_metadata(spec: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import scipy

    blas = np.__config__.CONFIG.get("Build Dependencies", {}).get("blas", {})
    return {
        "creation_timestamp_utc": spec["creation_timestamp_utc"],
        "source_commit": state["commit"],
        "source_short_commit": state["short_commit"],
        "source_branch": state["branch"],
        "source_bundle_scope": "committed_source_only",
        "uncommitted_source_changes_included": False,
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.mac_ver()[0],
            "machine": platform.machine(),
        },
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation()},
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas": {
            "name": blas.get("name"),
            "version": blas.get("version"),
            "configuration": blas.get("openblas configuration"),
        },
    }


def _readme(kind: str, spec: Mapping[str, Any]) -> str:
    commit = spec["source"]["commit"]
    if kind == "core":
        return f"""# One-dimensional reproduction archive

This core archive preserves the validated sigmoid-benchmark scientific inputs,
authoritative Figure 1--5 bundles, final presentation figures, compact final
case provenance, and a Git bundle containing source commit `{commit}`.

It supports inspection and rerunning final cases without recomputing the FOM,
derivatives, or POD/SVD. Figure 4 nonlinear cases use the tracked regenerated
parameters; they are not recovered historical manuscript parameters.

Verify the archive before use:

```bash
python scripts/1d/verify_reproduction_archive.py <core-archive>
```

Restore source with `git clone source/repository.bundle source-checkout`, then
restore `scientific_inputs/` and use the paths documented in
`provenance/publication_workflow.md`. The audit supplement is optional for
ordinary reruns and is required only to inspect the full Phase 8 search and
earlier diagnostic history.

The approved relative space-time metric, solve-only online timing, localized
sigmoid provenance, and historical limitations are preserved in `provenance/`.
This archive does not claim exact historical reproduction.
"""
    return f"""# One-dimensional audit supplement

This supplement contains Phase 5/5B reports, Phase 7 planning records, the full
Phase 8 Figure 4 search definition/index/candidate records, rejected-candidate
diagnostics, earlier diagnostic plots, and superseded Figure 4/5 renders.

It intentionally omits the production snapshot, derivatives, POD arrays,
authoritative bundles, final cases, and final figures. Those files are in the
core archive for source commit `{commit}`. This supplement depends on that core
archive and cannot rerun the scientific cases by itself.

Verify it with:

```bash
python scripts/1d/verify_reproduction_archive.py <audit-archive>
```

The search records document regenerated sigmoid selections, not recovered
historical manuscript parameters.
"""


def _copy_payload(
    entries: Iterable[ArchiveEntry],
    archive_root: Path,
    repository_root: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    metadata: dict[str, dict[str, Any]] = {}
    rewrites: list[dict[str, Any]] = []
    for entry in entries:
        destination = archive_root / entry.archive_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        source_sha = sha256_file(entry.source)
        data, transformed = _portable_bytes(entry.source, repository_root)
        destination.write_bytes(data)
        archive_sha = sha256_bytes(data)
        metadata[entry.archive_path] = {
            "original_repository_path": entry.source_relative,
            "scientific_role": entry.role,
            "tracked_status": entry.tracked_status,
            "source_run_id": entry.run_id,
            "authority": entry.authority,
            "source_sha256": source_sha,
            "portable_path_rewrite": transformed,
        }
        if transformed:
            rewrites.append(
                {
                    "archive_path": entry.archive_path,
                    "original_repository_path": entry.source_relative,
                    "source_sha256": source_sha,
                    "archive_sha256": archive_sha,
                }
            )
    return metadata, rewrites


def _add_generated_metadata(
    metadata: dict[str, dict[str, Any]],
    path: str,
    role: str,
    authority: str,
) -> None:
    metadata[path] = {
        "original_repository_path": None,
        "scientific_role": role,
        "tracked_status": "generated",
        "source_run_id": None,
        "authority": authority,
        "source_sha256": None,
        "portable_path_rewrite": False,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _inventory_records(
    archive_root: Path,
    metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for relative, details in sorted(metadata.items()):
        if not is_safe_relative_path(relative):
            raise ValueError(f"unsafe inventory path: {relative}")
        path = archive_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"inventory payload is missing: {relative}")
        records.append(
            {
                "archive_path": relative,
                "original_repository_path": details["original_repository_path"],
                "scientific_role": details["scientific_role"],
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "source_sha256": details["source_sha256"],
                "tracked_status": details["tracked_status"],
                "source_run_id": details["source_run_id"],
                "authority": details["authority"],
                "portable_path_rewrite": details["portable_path_rewrite"],
            }
        )
    return records


def _write_inventory_tsv(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    fields = [
        "archive_path",
        "original_repository_path",
        "scientific_role",
        "size_bytes",
        "sha256",
        "source_sha256",
        "tracked_status",
        "source_run_id",
        "authority",
        "portable_path_rewrite",
    ]
    lines = ["\t".join(fields)]
    for record in records:
        lines.append(
            "\t".join(
                "" if record[field] is None else str(record[field]).replace("\t", " ")
                for field in fields
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _important_archive_records(
    spec: Mapping[str, Any],
    runs: Mapping[str, str],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    by_source = {
        record["original_repository_path"]: record
        for record in records
        if record["original_repository_path"] is not None
    }
    result: dict[str, Any] = {}
    for important in spec.get("important_files", []):
        source = _format(important["path"], runs)
        record = by_source.get(source)
        if record is None:
            raise ValueError(f"important file is absent from core archive: {source}")
        result[important["name"]] = {
            "archive_path": record["archive_path"],
            "archive_sha256": record["sha256"],
            "original_repository_path": source,
            "original_sha256": important["sha256"],
            "portable_path_rewrite": record["portable_path_rewrite"],
        }
    return result


def _write_controls(
    archive_root: Path,
    *,
    kind: str,
    spec: Mapping[str, Any],
    runs: Mapping[str, str],
    metadata: dict[str, dict[str, Any]],
    validation: Mapping[str, Any],
    archive_format: str,
) -> dict[str, Any]:
    records = _inventory_records(archive_root, metadata)
    inventory = {
        "schema_version": "1.0.0",
        "archive_kind": kind,
        "archive_root": ARCHIVE_ROOTS[kind],
        "ordering": "archive_path_ascending",
        "control_files_excluded_to_avoid_self_reference": sorted(CONTROL_FILES),
        "entries": records,
    }
    inventory_path = archive_root / "inventory.json"
    _write_json(inventory_path, inventory)
    tsv_path = archive_root / "inventory.tsv"
    _write_inventory_tsv(tsv_path, records)
    inventory_checksum = sha256_file(inventory_path)
    archive_metadata = {
        "schema_version": "1.0.0",
        "archive_kind": kind,
        "archive_root": ARCHIVE_ROOTS[kind],
        "creation_timestamp_utc": spec["creation_timestamp_utc"],
        "archive_date": spec["archive_date"],
        "format": archive_format,
        "source": dict(spec["source"]),
        "source_bundle_scope": "committed_source_only",
        "uncommitted_source_changes_included": False,
        "payload_file_count": len(records),
        "payload_uncompressed_size_bytes": sum(record["size_bytes"] for record in records),
        "inventory_checksum_sha256": inventory_checksum,
        "inventory_tsv_checksum_sha256": sha256_file(tsv_path),
        "core_dependency": (
            None
            if kind == "core"
            else _archive_name("core", spec)
        ),
        "scientific_checksums": dict(spec.get("scientific_checksums", {})),
        "important_files": (
            _important_archive_records(spec, runs, records) if kind == "core" else {}
        ),
        "validation_summary": {
            "authoritative_input_count": len(validation.get("validated_files", {})),
            "shared_array_count": len(validation.get("shared_arrays", {})),
            "bundle_count": len(validation.get("bundles", {})),
            "final_figure_count": len(validation.get("final_figures", {})),
        },
        "archive_sha256_recorded_in_external_sidecar": True,
    }
    metadata_path = archive_root / "archive_metadata.json"
    _write_json(metadata_path, archive_metadata)
    sums: list[str] = []
    for path in sorted(archive_root.rglob("*"), key=lambda item: item.relative_to(archive_root).as_posix()):
        if path.is_file() and path.name != "SHA256SUMS":
            relative = path.relative_to(archive_root).as_posix()
            sums.append(f"{sha256_file(path)}  {relative}")
    (archive_root / "SHA256SUMS").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return archive_metadata


def _write_tar_members(archive: tarfile.TarFile, source_root: Path) -> None:
    for path in sorted(
        source_root.rglob("*"),
        key=lambda item: item.relative_to(source_root.parent).as_posix(),
    ):
        if not path.is_file():
            continue
        name = path.relative_to(source_root.parent).as_posix()
        if not is_safe_relative_path(name):
            raise ValueError(f"unsafe tar member name: {name}")
        info = tarfile.TarInfo(name=name)
        info.size = path.stat().st_size
        info.mode = 0o644
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        with path.open("rb") as stream:
            archive.addfile(info, stream)


def preferred_archive_format() -> tuple[str, str]:
    if shutil.which("zstd") is not None:
        return "tar.zst", "deterministic_tar_zstandard"
    return "tar.gz", "deterministic_tar_gzip"


def create_deterministic_archive(source_root: Path, archive_path: Path) -> None:
    if archive_path.exists():
        raise FileExistsError(f"refusing to overwrite archive: {archive_path}")
    temporary = archive_path.with_name("." + archive_path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary archive path already exists: {temporary}")
    try:
        if archive_path.name.endswith(".tar.zst"):
            executable = shutil.which("zstd")
            if executable is None:
                raise RuntimeError("zstd is required to create a .tar.zst archive")
            with temporary.open("wb") as output:
                process = subprocess.Popen(
                    [
                        executable,
                        "--compress",
                        "--stdout",
                        "--quiet",
                        "--threads=1",
                        "-3",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=output,
                    stderr=subprocess.PIPE,
                )
                if process.stdin is None:
                    raise RuntimeError("could not open zstd compression stream")
                try:
                    with tarfile.open(
                        fileobj=process.stdin,
                        mode="w|",
                        format=tarfile.GNU_FORMAT,
                    ) as archive:
                        _write_tar_members(archive, source_root)
                    process.stdin.close()
                    stderr = process.stderr.read() if process.stderr is not None else b""
                    return_code = process.wait(timeout=300)
                    if return_code != 0:
                        raise RuntimeError(
                            "zstd compression failed: "
                            + stderr.decode("utf-8", errors="replace").strip()
                        )
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
        elif archive_path.name.endswith(".tar.gz"):
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    compresslevel=6,
                    mtime=0,
                ) as compressed:
                    with tarfile.open(
                        fileobj=compressed,
                        mode="w",
                        format=tarfile.GNU_FORMAT,
                    ) as archive:
                        _write_tar_members(archive, source_root)
        else:
            raise ValueError(f"unsupported archive suffix: {archive_path.name}")
        os.replace(temporary, archive_path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _archive_name(kind: str, spec: Mapping[str, Any]) -> str:
    suffix, _ = preferred_archive_format()
    return (
        f"nonlinear-manifolds-1d-{kind}-{spec['source']['short_commit']}-"
        f"{spec['archive_date']}.{suffix}"
    )


def plan_archives(
    spec: Mapping[str, Any],
    *,
    repository_root: str | Path = REPOSITORY_ROOT,
    output_directory: str | Path = "dist/1d",
    run_overrides: Mapping[str, str] | None = None,
    allow_unknown_run_ids: bool = False,
) -> dict[str, Any]:
    root = Path(repository_root).resolve()
    validation = validate_authoritative_inputs(
        spec,
        repository_root=root,
        run_overrides=run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    core = collect_archive_entries(
        spec,
        "core",
        repository_root=root,
        run_overrides=run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    audit = collect_archive_entries(
        spec,
        "audit",
        repository_root=root,
        run_overrides=run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    assert_core_audit_non_overlap(core, audit)
    core_bytes = sum(entry.source.stat().st_size for entry in core)
    audit_bytes = sum(entry.source.stat().st_size for entry in audit)
    available = shutil.disk_usage(root).free
    estimated_temporary = 2 * (core_bytes + audit_bytes)
    if available < estimated_temporary:
        raise OSError(
            f"insufficient free space: need about {estimated_temporary} bytes, have {available}"
        )
    output = Path(output_directory)
    return {
        "action": "would_build_reproduction_archives",
        "writes_files": False,
        "launches_scientific_execution": False,
        "format": preferred_archive_format()[0],
        "source": validation["source"],
        "available_disk_bytes": available,
        "estimated_temporary_space_bytes": estimated_temporary,
        "archives": {
            "core": {
                "path": str(output / _archive_name("core", spec)),
                "source_file_count": len(core),
                "estimated_uncompressed_bytes": core_bytes,
            },
            "audit": {
                "path": str(output / _archive_name("audit", spec)),
                "source_file_count": len(audit),
                "estimated_uncompressed_bytes": audit_bytes,
            },
        },
        "validation": validation,
    }


def _build_one(
    kind: str,
    spec: Mapping[str, Any],
    entries: list[ArchiveEntry],
    validation: Mapping[str, Any],
    *,
    repository_root: Path,
    output_directory: Path,
    runs: Mapping[str, str],
    overwrite: bool,
) -> dict[str, Any]:
    archive_path = output_directory / _archive_name(kind, spec)
    sidecar = Path(str(archive_path) + ".sha256")
    if not overwrite and (archive_path.exists() or sidecar.exists()):
        raise FileExistsError(f"refusing to overwrite archive output: {archive_path}")
    if overwrite:
        for path in (archive_path, sidecar):
            if path.exists():
                path.unlink()
    staging = Path(tempfile.mkdtemp(prefix=f".staging-{kind}-", dir=output_directory))
    archive_root = staging / ARCHIVE_ROOTS[kind]
    archive_root.mkdir()
    try:
        metadata, rewrites = _copy_payload(entries, archive_root, repository_root)
        readme_path = archive_root / "README.md"
        readme_path.write_text(_readme(kind, spec), encoding="utf-8")
        _add_generated_metadata(metadata, "README.md", "archive guide", "authoritative" if kind == "core" else "supplemental")
        manifests = archive_root / "manifests"
        manifests.mkdir(exist_ok=True)
        path_map = manifests / "portable_path_map.json"
        _write_json(
            path_map,
            {
                "schema_version": "1.0.0",
                "policy": "archive copies replace repository-local absolute paths with relative paths; source files are unchanged",
                "rewritten_files": rewrites,
            },
        )
        _add_generated_metadata(
            metadata,
            "manifests/portable_path_map.json",
            "portable path rewrite map",
            "authoritative" if kind == "core" else "supplemental",
        )
        if kind == "core":
            source_dir = archive_root / "source"
            source_dir.mkdir(exist_ok=True)
            environment = source_dir / "source_environment.json"
            _write_json(environment, _environment_metadata(spec, validation["source"]))
            _add_generated_metadata(metadata, "source/source_environment.json", "source and environment metadata", "authoritative")
            bundle = source_dir / "repository.bundle"
            subprocess.run(
                ["git", "bundle", "create", str(bundle), "HEAD"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                timeout=300,
            )
            bundle_heads = subprocess.run(
                ["git", "bundle", "list-heads", str(bundle)],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            ).stdout
            if spec["source"]["commit"] not in bundle_heads:
                raise ValueError("Git bundle does not advertise the pinned source commit")
            subprocess.run(
                ["git", "bundle", "verify", str(bundle)],
                cwd=repository_root,
                check=True,
                capture_output=True,
                timeout=300,
            )
            _add_generated_metadata(metadata, "source/repository.bundle", "exact-source Git bundle", "authoritative")
        archive_metadata = _write_controls(
            archive_root,
            kind=kind,
            spec=spec,
            runs=runs,
            metadata=metadata,
            validation=validation,
            archive_format=preferred_archive_format()[1],
        )
        create_deterministic_archive(archive_root, archive_path)
        archive_sha = sha256_file(archive_path)
        sidecar.write_text(f"{archive_sha}  {archive_path.name}\n", encoding="utf-8")
        return {
            "kind": kind,
            "path": str(archive_path),
            "sha256": archive_sha,
            "compressed_size_bytes": archive_path.stat().st_size,
            "payload_file_count": archive_metadata["payload_file_count"],
            "payload_uncompressed_size_bytes": archive_metadata[
                "payload_uncompressed_size_bytes"
            ],
            "inventory_checksum_sha256": archive_metadata[
                "inventory_checksum_sha256"
            ],
            "sidecar": str(sidecar),
        }
    finally:
        shutil.rmtree(staging)


def build_archives(
    spec: Mapping[str, Any],
    *,
    kind: str = "both",
    repository_root: str | Path = REPOSITORY_ROOT,
    output_directory: str | Path = "dist/1d",
    dry_run: bool = False,
    overwrite: bool = False,
    run_overrides: Mapping[str, str] | None = None,
    allow_unknown_run_ids: bool = False,
) -> dict[str, Any]:
    if kind not in {"core", "audit", "both"}:
        raise ValueError("kind must be core, audit, or both")
    plan = plan_archives(
        spec,
        repository_root=repository_root,
        output_directory=output_directory,
        run_overrides=run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    if dry_run:
        return plan
    root = Path(repository_root).resolve()
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    runs = _resolved_runs(
        spec,
        run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    core = collect_archive_entries(
        spec,
        "core",
        repository_root=root,
        run_overrides=run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    audit = collect_archive_entries(
        spec,
        "audit",
        repository_root=root,
        run_overrides=run_overrides,
        allow_unknown_run_ids=allow_unknown_run_ids,
    )
    assert_core_audit_non_overlap(core, audit)
    validation = plan["validation"]
    selected_kinds = ("core", "audit") if kind == "both" else (kind,)
    results = []
    for selected in selected_kinds:
        results.append(
            _build_one(
                selected,
                spec,
                core if selected == "core" else audit,
                validation,
                repository_root=root,
                output_directory=output,
                runs=runs,
                overwrite=overwrite,
            )
        )
    return {
        "action": "built_reproduction_archives",
        "launches_scientific_execution": False,
        "archives": results,
    }


def _parse_sha256sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None or not is_safe_relative_path(match.group(2)):
            raise ValueError("invalid SHA256SUMS entry")
        if match.group(2) in result:
            raise ValueError("duplicate SHA256SUMS path")
        result[match.group(2)] = match.group(1)
    return result


def verify_extracted_tree(root: str | Path) -> dict[str, Any]:
    archive_root = Path(root)
    required = CONTROL_FILES | {"README.md"}
    missing_controls = sorted(name for name in required if not (archive_root / name).is_file())
    if missing_controls:
        raise ValueError("missing archive control files: " + ", ".join(missing_controls))
    expected = _parse_sha256sums(archive_root / "SHA256SUMS")
    actual_files = {
        path.relative_to(archive_root).as_posix()
        for path in archive_root.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    }
    missing = sorted(set(expected).difference(actual_files))
    extra = sorted(actual_files.difference(expected))
    if missing:
        raise ValueError("archive is missing files: " + ", ".join(missing[:5]))
    if extra:
        raise ValueError("archive contains extra files: " + ", ".join(extra[:5]))
    for relative, checksum in sorted(expected.items()):
        if sha256_file(archive_root / relative) != checksum:
            raise ValueError(f"archive file checksum mismatch: {relative}")
    inventory_path = archive_root / "inventory.json"
    inventory = _load_json(inventory_path)
    records = inventory.get("entries")
    if not isinstance(records, list):
        raise ValueError("archive inventory entries are invalid")
    paths = [record.get("archive_path") for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("archive inventory ordering or uniqueness is invalid")
    payload_expected = actual_files.difference(CONTROL_FILES)
    if set(paths) != payload_expected:
        raise ValueError("archive inventory has missing or extra payload entries")
    for record in records:
        relative = record["archive_path"]
        if not is_safe_relative_path(relative):
            raise ValueError("archive inventory contains unsafe path")
        original = record.get("original_repository_path")
        if original is not None and not is_safe_relative_path(original):
            raise ValueError("archive inventory contains absolute original path")
        path = archive_root / relative
        if path.stat().st_size != record["size_bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"archive inventory disagrees with payload: {relative}")
    metadata = _load_json(archive_root / "archive_metadata.json")
    inventory_checksum = sha256_file(inventory_path)
    if metadata.get("inventory_checksum_sha256") != inventory_checksum:
        raise ValueError("archive inventory checksum does not match metadata")
    for name, record in sorted(metadata.get("important_files", {}).items()):
        relative = record.get("archive_path")
        if not is_safe_relative_path(relative) or sha256_file(archive_root / relative) != record.get(
            "archive_sha256"
        ):
            raise ValueError(f"important scientific checksum failed: {name}")
    all_files = {
        path.relative_to(archive_root).as_posix(): path.stat().st_size
        for path in archive_root.rglob("*")
        if path.is_file()
    }
    return {
        "archive_kind": metadata["archive_kind"],
        "archive_root": metadata["archive_root"],
        "verified_file_count": len(all_files),
        "uncompressed_content_size_bytes": sum(all_files.values()),
        "inventory_checksum_sha256": inventory_checksum,
        "important_scientific_files_verified": len(metadata.get("important_files", {})),
        "missing_files": [],
        "extra_files": [],
    }


def _safe_extract(archive_path: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    roots: set[str] = set()
    with tarfile.open(archive_path, mode="r:*") as archive:
        members = archive.getmembers()
        for member in members:
            if not is_safe_relative_path(member.name):
                raise ValueError(f"unsafe archive member path: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsafe archive member type: {member.name}")
            if not member.isfile() and not member.isdir():
                raise ValueError(f"unsupported archive member type: {member.name}")
            roots.add(PurePosixPath(member.name).parts[0])
        if len(roots) != 1:
            raise ValueError("archive must contain exactly one top-level directory")
        for member in members:
            target = destination / member.name
            resolved = target.resolve()
            try:
                resolved.relative_to(destination.resolve())
            except ValueError:
                raise ValueError(f"archive extraction target escapes destination: {member.name}") from None
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return destination / next(iter(roots))


def _decompress_zstandard_archive(archive_path: Path) -> Path:
    executable = shutil.which("zstd")
    if executable is None:
        raise RuntimeError("zstd is required to verify a .tar.zst archive")
    descriptor, temporary_name = tempfile.mkstemp(prefix="verify-1d-zstd-", suffix=".tar")
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as output:
            completed = subprocess.run(
                [executable, "--decompress", "--stdout", "--quiet", str(archive_path)],
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=300,
            )
        if completed.returncode != 0:
            raise RuntimeError(
                "zstd decompression failed: "
                + completed.stderr.decode("utf-8", errors="replace").strip()
            )
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def verify_archive(
    archive: str | Path,
    *,
    expected_sha256: str | None = None,
    extract_to: str | Path | None = None,
) -> dict[str, Any]:
    archive_path = Path(archive)
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive is missing: {archive_path}")
    actual_archive_sha = sha256_file(archive_path)
    sidecar = Path(str(archive_path) + ".sha256")
    if expected_sha256 is None and sidecar.is_file():
        line = sidecar.read_text(encoding="utf-8").strip()
        match = re.fullmatch(r"([0-9a-f]{64})  ([^/]+)", line)
        if match is None or match.group(2) != archive_path.name:
            raise ValueError("archive SHA-256 sidecar is invalid")
        expected_sha256 = match.group(1)
    if expected_sha256 is not None and actual_archive_sha != expected_sha256:
        raise ValueError("archive checksum mismatch")
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if extract_to is None:
        temporary = tempfile.TemporaryDirectory(prefix="verify-1d-archive-")
        destination = Path(temporary.name) / "extract"
    else:
        destination = Path(extract_to)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite extraction directory: {destination}")
    decompressed_tar: Path | None = None
    try:
        tar_source = archive_path
        if archive_path.name.endswith(".tar.zst"):
            decompressed_tar = _decompress_zstandard_archive(archive_path)
            tar_source = decompressed_tar
        root = _safe_extract(tar_source, destination)
        report = verify_extracted_tree(root)
        report.update(
            {
                "archive_path": str(archive_path),
                "archive_sha256": actual_archive_sha,
                "compressed_size_bytes": archive_path.stat().st_size,
                "safe_extraction": True,
                "extraction_path": None if temporary is not None else str(root),
            }
        )
        return report
    finally:
        if decompressed_tar is not None:
            decompressed_tar.unlink(missing_ok=True)
        if temporary is not None:
            temporary.cleanup()


def create_synthetic_archive(
    output: str | Path,
    files: Mapping[str, bytes],
    *,
    root_name: str = "synthetic_archive",
) -> dict[str, Any]:
    """Test helper for archive-format validation; never touches production data."""
    if not is_safe_relative_path(root_name):
        raise ValueError("unsafe synthetic archive root")
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="synthetic-archive-"))
    root = staging / root_name
    root.mkdir()
    metadata: dict[str, dict[str, Any]] = {}
    try:
        for relative, content in sorted(files.items()):
            if not is_safe_relative_path(relative) or relative in CONTROL_FILES:
                raise ValueError(f"unsafe synthetic archive path: {relative}")
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            _add_generated_metadata(metadata, relative, "synthetic test payload", "supplemental")
        if "README.md" not in metadata:
            (root / "README.md").write_text("# Synthetic archive\n", encoding="utf-8")
            _add_generated_metadata(metadata, "README.md", "archive guide", "supplemental")
        spec = {
            "creation_timestamp_utc": "2000-01-01T00:00:00Z",
            "archive_date": "20000101",
            "source": {"commit": "0" * 40, "short_commit": "0000000", "branch": "test"},
            "scientific_checksums": {},
            "important_files": [],
        }
        _write_controls(
            root,
            kind="audit",
            spec=spec,
            runs={},
            metadata=metadata,
            validation={},
            archive_format="deterministic_tar_gzip",
        )
        create_deterministic_archive(root, output_path)
        checksum = sha256_file(output_path)
        Path(str(output_path) + ".sha256").write_text(
            f"{checksum}  {output_path.name}\n", encoding="utf-8"
        )
        return {"path": str(output_path), "sha256": checksum}
    finally:
        shutil.rmtree(staging)
