"""Safe orchestration for configured one-dimensional FOM and ROM stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time as wall_time
from typing import Any

import numpy as np

from .config import OneDConfig
from .fom import inspect_snapshot, save_snapshot, solve_fom
from .provenance import create_run_directory, sha256_file, update_manifest
from .rom import normalize_model_name, run_selected_rom


@dataclass(frozen=True)
class FomDryRunReport:
    configuration: str
    configuration_checksum: str
    angular_ordinates: int
    cells: int
    spatial_dofs: int
    phase_space_dofs: int
    output_times: int
    expected_snapshot_shape: tuple[int, int]
    expected_snapshot_path: str
    estimated_raw_snapshot_bytes: int
    estimated_raw_snapshot_mib: float
    snapshot_exists: bool
    snapshot_compatible: bool | None
    action: str
    assembles_operators: bool = False
    solves: bool = False
    writes_files: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RomDryRunReport:
    configuration: str
    configuration_checksum: str
    model: str
    operators: str
    latent_dimension: int
    lifting_dimension: int
    training_end_time: float
    input_snapshot_path: str
    expected_snapshot_shape: tuple[int, int]
    snapshot_exists: bool
    snapshot_compatible: bool | None
    action: str
    historical_full_sequence: bool
    assembles_operators: bool = False
    solves: bool = False
    writes_files: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _configured_snapshot_path(
    config: OneDConfig,
    *,
    snapshot_path: str | Path | None = None,
    output_directory: str | Path | None = None,
) -> Path:
    if snapshot_path is not None:
        return Path(snapshot_path)
    if output_directory is not None:
        return Path(output_directory) / "data" / config.output.snapshot_filename
    return Path(config.output.snapshot_filename)


def dry_run_fom(
    config: OneDConfig,
    *,
    snapshot_path: str | Path | None = None,
    output_directory: str | Path | None = None,
    overwrite: bool = False,
) -> FomDryRunReport:
    """Report FOM dimensions and file decisions without assembly or writes."""
    path = _configured_snapshot_path(
        config, snapshot_path=snapshot_path, output_directory=output_directory
    )
    inspection = inspect_snapshot(path, config) if path.exists() else None
    if inspection is None:
        action = "would_solve"
        compatible = None
    elif not inspection.compatible:
        action = "refuse_invalid_snapshot"
        compatible = False
    elif config.output.reuse_existing_snapshot and not overwrite:
        action = "reuse"
        compatible = True
    elif overwrite or config.output.allow_overwrite:
        action = "would_overwrite"
        compatible = True
    else:
        action = "refuse_existing_snapshot"
        compatible = True
    return FomDryRunReport(
        configuration=config.name,
        configuration_checksum=config.checksum(),
        angular_ordinates=config.problem.angular_ordinates,
        cells=config.problem.cell_count,
        spatial_dofs=config.problem.spatial_dofs,
        phase_space_dofs=config.problem.phase_space_dofs,
        output_times=config.time.output_count,
        expected_snapshot_shape=config.expected_snapshot_shape,
        expected_snapshot_path=str(path),
        estimated_raw_snapshot_bytes=config.expected_snapshot_bytes_float64,
        estimated_raw_snapshot_mib=(
            config.expected_snapshot_bytes_float64 / (1024.0**2)
        ),
        snapshot_exists=path.exists(),
        snapshot_compatible=compatible,
        action=action,
    )


def dry_run_rom(
    config: OneDConfig,
    *,
    model: str,
    operators: str,
    input_snapshot: str | Path | None = None,
    historical_full_sequence: bool = False,
) -> RomDryRunReport:
    """Report one ROM selection and input decision without scientific work."""
    model = normalize_model_name(model)
    if operators not in {"projected", "inferred"}:
        raise ValueError("operators must be 'projected' or 'inferred'")
    path = Path(input_snapshot or config.output.snapshot_filename)
    inspection = inspect_snapshot(path, config) if path.exists() else None
    if inspection is None:
        action = "refuse_missing_snapshot"
        compatible = None
    elif not inspection.compatible:
        action = "refuse_invalid_snapshot"
        compatible = False
    elif historical_full_sequence:
        action = "would_run_historical_sequence"
        compatible = True
    else:
        action = "would_run_selected_model"
        compatible = True
    return RomDryRunReport(
        configuration=config.name,
        configuration_checksum=config.checksum(),
        model=model,
        operators=operators,
        latent_dimension=config.rom.latent_dimension,
        lifting_dimension=config.rom.lifting_dimension,
        training_end_time=config.time.training_end_time,
        input_snapshot_path=str(path),
        expected_snapshot_shape=config.expected_snapshot_shape,
        snapshot_exists=path.exists(),
        snapshot_compatible=compatible,
        action=action,
        historical_full_sequence=historical_full_sequence,
    )


def execute_fom_workflow(
    config: OneDConfig,
    *,
    execute: bool = False,
    run_directory: str | Path | None = None,
    config_source: str | Path | None = None,
    existing_snapshot: str | Path | None = None,
    overwrite: bool = False,
    hash_snapshot: bool = False,
) -> dict[str, Any]:
    """Execute or reuse one FOM snapshot only after explicit authorization."""
    if not execute:
        raise PermissionError("FOM execution requires execute=True")
    run = create_run_directory(
        config,
        run_directory=run_directory,
        config_source=config_source,
        execution_stage="fom",
    )
    target = run.data / config.output.snapshot_filename
    source_path = Path(existing_snapshot) if existing_snapshot else target
    started = datetime.now(timezone.utc)
    timer = wall_time.perf_counter()
    update_manifest(
        run,
        execution={"start_time_utc": started.isoformat(), "stage": "fom"},
    )

    inspection = inspect_snapshot(source_path, config) if source_path.exists() else None
    if inspection and inspection.compatible and config.output.reuse_existing_snapshot and not overwrite:
        elapsed = wall_time.perf_counter() - timer
        update_manifest(
            run,
            snapshot={
                "filename": source_path.name,
                "shape": list(inspection.shape),
                "dtype": inspection.dtype,
                "content_sha256": sha256_file(source_path) if hash_snapshot else None,
            },
            execution={
                "solver_success": None,
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "diagnostics": {"action": "reused", "source": str(source_path)},
            },
        )
        return {"action": "reused", "snapshot_path": source_path, "run": run}
    if inspection and not inspection.compatible:
        raise ValueError(
            "existing snapshot is incompatible: "
            + "; ".join(inspection.compatibility_errors)
        )
    if source_path.exists() and not (overwrite or config.output.allow_overwrite):
        raise FileExistsError(f"refusing to overwrite existing snapshot: {source_path}")

    try:
        solution = solve_fom(config)
    except Exception as error:
        elapsed = wall_time.perf_counter() - timer
        update_manifest(
            run,
            execution={
                "solver_success": False,
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "diagnostics": {
                    "action": "failed",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            },
        )
        raise
    save_snapshot(
        target,
        solution.state,
        config,
        overwrite=overwrite or config.output.allow_overwrite,
    )
    elapsed = wall_time.perf_counter() - timer
    solver_result = solution.solver_result
    snapshot_hash = sha256_file(target) if hash_snapshot else None
    diagnostics = {
        "action": "solved",
        "solver_message": str(solver_result.message),
        "output_time_count": int(solution.time.size),
        "final_time": float(solution.time[-1]),
        "expected_final_time": float(config.time.final_time),
        "final_time_confirmed": bool(
            np.isclose(
                solution.time[-1],
                config.time.final_time,
                rtol=0.0,
                atol=1.0e-12,
            )
        ),
        "rhs_evaluations": int(getattr(solver_result, "nfev", 0)),
        "jacobian_evaluations": int(getattr(solver_result, "njev", 0)),
        "lu_decompositions": int(getattr(solver_result, "nlu", 0)),
        "snapshot_minimum": float(np.min(solution.state)),
        "snapshot_maximum": float(np.max(solution.state)),
        "snapshot_finite": bool(np.all(np.isfinite(solution.state))),
        "snapshot_file_bytes": int(target.stat().st_size),
    }
    update_manifest(
        run,
        snapshot={
            "filename": target.name,
            "shape": list(solution.state.shape),
            "dtype": str(solution.state.dtype),
            "content_sha256": snapshot_hash,
        },
        execution={
            "solver_success": bool(solver_result.success),
            "finish_time_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "diagnostics": diagnostics,
        },
    )
    return {"action": "solved", "snapshot_path": target, "run": run}


def execute_rom_workflow(
    config: OneDConfig,
    *,
    model: str,
    operators: str,
    input_snapshot: str | Path,
    execute: bool = False,
    run_directory: str | Path | None = None,
    config_source: str | Path | None = None,
) -> dict[str, Any]:
    """Execute one selected ROM case only after explicit authorization."""
    if not execute:
        raise PermissionError("ROM execution requires execute=True")
    inspection = inspect_snapshot(input_snapshot, config)
    if not inspection.compatible:
        raise ValueError(
            "input snapshot is incompatible: "
            + "; ".join(inspection.compatibility_errors)
        )
    run = create_run_directory(
        config,
        run_directory=run_directory,
        config_source=config_source,
        execution_stage="rom",
        parent_provenance={"input_snapshot": str(input_snapshot)},
    )
    started = datetime.now(timezone.utc)
    timer = wall_time.perf_counter()
    update_manifest(
        run,
        execution={"start_time_utc": started.isoformat(), "stage": "rom"},
    )
    result = run_selected_rom(
        config,
        str(input_snapshot),
        model_name=model,
        operator_choice=operators,
    )
    np.save(run.data / "reduced_state.npy", result.reduced_state)
    np.save(run.data / "reconstructed_state.npy", result.reconstructed_state)
    metrics = {
        "model": result.model_name,
        "operators": result.operator_choice,
        "maximum_error": float(np.max(result.errors)),
        "mean_error": float(np.mean(result.errors)),
        "diagnostics": result.diagnostics,
    }
    (run.metrics / "summary.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    elapsed = wall_time.perf_counter() - timer
    update_manifest(
        run,
        snapshot={
            "filename": Path(input_snapshot).name,
            "shape": list(inspection.shape),
            "dtype": inspection.dtype,
            "content_sha256": None,
        },
        execution={
            "solver_success": bool(result.diagnostics["solver_success"]),
            "finish_time_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed,
            "diagnostics": metrics,
        },
    )
    return {"result": result, "run": run, "metrics": metrics}
