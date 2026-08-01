"""Validated JSON/NPZ artifacts for sigmoid-benchmark publication cases."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import scipy

from .config import OneDConfig
from .publication_experiments import (
    BENCHMARK_VARIANT,
    EXPECTED_DEVIATION,
    PublicationCase,
    PublicationCatalog,
    resolve_base_configuration,
    resolve_case_configuration,
)
from .publication_metrics import (
    publication_metric_definitions,
    pod_energy_curves,
    validate_timing_metadata,
)


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
FIGURE4_EXPECTED_SERIES = {
    f"{operators}_{model}"
    for operators in ("projected", "inferred")
    for model in ("linear", "elementwise", "tensorial")
}
FIGURE_EXPECTED_CASES = {
    "Figure 1": {"fig1_pod_reducibility"},
    "Figure 2": {
        "fig2_linear_projected",
        "fig2_elementwise_projected",
        "fig2_tensorial_projected",
    },
    "Figure 3": {
        "fig3_linear_inferred",
        "fig3_elementwise_inferred",
        "fig3_tensorial_inferred",
    },
    "Figure 4": {
        f"fig4_{model}_{operators}_nr{rank}"
        for model in ("linear", "elementwise", "tensorial")
        for operators in ("projected", "inferred")
        for rank in (8, 16, 24, 32, 40, 48, 56, 64)
    },
    "Figure 5": {"fig5_projected_nq_sweep", "fig5_inferred_nq_sweep"},
}


class PublicationExecutionRefused(RuntimeError):
    """Raised before writes when a publication case is not execution-ready."""


@dataclass(frozen=True)
class PublicationRunDirectory:
    root: Path
    config_path: Path
    case_path: Path
    manifest_path: Path
    metrics_path: Path
    diagnostics_path: Path
    data: Path
    figures: Path


@dataclass(frozen=True)
class FigureDataBundle:
    figure: str
    root: Path
    metadata_path: Path
    data_path: Path
    case_ids: tuple[str, ...]
    complete: bool


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON-serialize {type(value).__name__}")


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


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return data


def _git_metadata(repository_root: str | Path | None = None) -> dict[str, Any]:
    cwd = Path(repository_root or Path.cwd())
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


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _default_timing_metadata() -> dict[str, Any]:
    return {
        "online_runtime_seconds": None,
        "offline_runtime_seconds": None,
        "total_runtime_seconds": None,
        "speedup_basis": None,
        "included_stages": [],
        "excluded_stages": [],
        "classification": "not_yet_measured",
    }


def create_publication_run_directory(
    catalog: PublicationCatalog,
    case: PublicationCase,
    config: OneDConfig,
    *,
    input_snapshot: str | Path,
    input_snapshot_checksum: str,
    input_snapshot_shape: tuple[int, int] | None = None,
    input_snapshot_dtype: str = "float64",
    run_id: str | None = None,
    output_root: str | Path = "results/1d/publication",
    run_directory: str | Path | None = None,
    repository_root: str | Path | None = None,
) -> PublicationRunDirectory:
    """Create the immutable publication layout after execution is authorized."""
    created = datetime.now(timezone.utc)
    run_id = run_id or created.strftime("%Y%m%dT%H%M%SZ") + "-" + catalog.checksum()[:8]
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id contains unsupported characters")
    root = (
        Path(run_directory)
        if run_directory is not None
        else Path(output_root) / case.case_id / run_id
    )
    if root.exists():
        raise FileExistsError(
            f"publication run directory already exists and is immutable: {root}"
        )
    data = root / "data"
    figures = root / "figures"
    data.mkdir(parents=True, exist_ok=False)
    figures.mkdir(exist_ok=False)
    run = PublicationRunDirectory(
        root=root,
        config_path=root / "config.json",
        case_path=root / "case.json",
        manifest_path=root / "manifest.json",
        metrics_path=root / "metrics.json",
        diagnostics_path=root / "diagnostics.json",
        data=data,
        figures=figures,
    )
    _write_json(run.config_path, config.to_dict())
    _write_json(run.case_path, case.to_dict())
    _write_json(run.metrics_path, {"schema_version": "1.0.0", "metrics": {}})
    _write_json(run.diagnostics_path, {"schema_version": "1.0.0", "diagnostics": {}})
    manifest = {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "case_id": case.case_id,
        "figure": case.figure,
        "creation_time_utc": created.isoformat(),
        "git": _git_metadata(repository_root),
        "runtime": {
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "platform": platform.platform(),
        },
        "experiment_catalog_path": str(catalog.source_path),
        "experiment_catalog_checksum": catalog.checksum(),
        "case_definition_checksum_sha256": hashlib.sha256(
            json.dumps(
                case.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest(),
        "resolved_configuration_checksum_sha256": config.checksum(),
        "base_configuration_path": case.base_configuration_path,
        "base_configuration_checksum": case.base_configuration_checksum,
        "benchmark_variant": case.benchmark_variant,
        "manuscript_deviation": case.manuscript_deviation,
        "input_snapshot": {
            "path": str(input_snapshot),
            "filename": Path(input_snapshot).name,
            "checksum_sha256": input_snapshot_checksum,
            "shape": list(input_snapshot_shape or config.expected_snapshot_shape),
            "dtype": input_snapshot_dtype,
        },
        "model": {
            "type": case.model_type,
            "operator_construction": case.operator_construction,
            "N_r": case.latent_dimension,
            "N_q": case.lifting_dimension,
            "regularization": {
                "gamma": case.lifting_regularization_gamma,
                "lambda_L": case.lambda_L,
                "lambda_Q": case.lambda_Q,
            },
            "inference_tolerance": case.inference_tolerance,
            "maximum_iterations": case.maximum_iterations,
        },
        "training_interval": list(case.training_interval),
        "output_times": {
            "initial": config.time.initial_time,
            "final": config.time.final_time,
            "spacing": config.time.output_spacing,
            "count": config.time.output_count,
        },
        "metric_definitions": publication_metric_definitions(),
        "timing": _default_timing_metadata(),
        "solver": {"status": "initialized", "success": None, "message": None},
        "arrays": [],
    }
    _write_json(run.manifest_path, manifest)
    return run


def _array_metadata(relative_path: str, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {
            "artifact_path": relative_path,
            "array_name": name,
            "shape": list(np.asarray(array).shape),
            "dtype": str(np.asarray(array).dtype),
        }
        for name, array in sorted(arrays.items())
    ]


def write_npz_artifact(
    run: PublicationRunDirectory,
    filename: str,
    arrays: dict[str, np.ndarray],
) -> Path:
    """Write compact, named arrays and append their shape/dtype manifest records."""
    if not filename.endswith(".npz") or Path(filename).name != filename:
        raise ValueError("NPZ artifact filename must be a simple .npz basename")
    if not arrays:
        raise ValueError("an NPZ artifact must contain at least one array")
    normalized = {name: np.asarray(value) for name, value in arrays.items()}
    for name, array in normalized.items():
        if not name or array.dtype == object:
            raise ValueError("artifact arrays require names and non-object dtypes")
    path = run.data / filename
    if path.exists():
        raise FileExistsError(f"refusing to overwrite publication artifact: {path}")
    np.savez_compressed(path, **normalized)
    manifest = _read_json(run.manifest_path)
    manifest["arrays"].extend(_array_metadata(f"data/{filename}", normalized))
    _write_json(run.manifest_path, manifest)
    return path


def update_publication_run(
    run: PublicationRunDirectory,
    *,
    metrics: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
    solver: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if metrics is not None:
        _write_json(run.metrics_path, {"schema_version": "1.0.0", "metrics": metrics})
    if diagnostics is not None:
        _write_json(
            run.diagnostics_path,
            {"schema_version": "1.0.0", "diagnostics": diagnostics},
        )
    manifest = _read_json(run.manifest_path)
    if solver is not None:
        manifest["solver"].update(solver)
    if timing is not None:
        validate_timing_metadata(timing)
        manifest["timing"] = timing
    _write_json(run.manifest_path, manifest)
    return manifest


def _validate_array_records(
    root: Path, records: list[dict[str, Any]]
) -> dict[tuple[str, str], np.ndarray]:
    loaded: dict[str, dict[str, np.ndarray]] = {}
    validated: dict[tuple[str, str], np.ndarray] = {}
    for record in records:
        required = {"artifact_path", "array_name", "shape", "dtype"}
        if set(record) != required:
            raise ValueError("array manifest records have invalid fields")
        relative = Path(record["artifact_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("array artifact paths must stay inside the run directory")
        path = root / relative
        if not path.is_file() or path.suffix != ".npz":
            raise ValueError(f"missing NPZ artifact: {path}")
        key = str(path)
        if key not in loaded:
            with np.load(path, allow_pickle=False) as archive:
                loaded[key] = {name: np.asarray(archive[name]) for name in archive.files}
        try:
            array = loaded[key][record["array_name"]]
        except KeyError:
            raise ValueError(f"missing array {record['array_name']} in {path}") from None
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"array metadata mismatch for {record['array_name']} in {path}")
        validated[(record["artifact_path"], record["array_name"])] = array
    return validated


def _validate_case_array_shapes(
    arrays: dict[tuple[str, str], np.ndarray],
    case: PublicationCase,
    config: OneDConfig,
) -> None:
    def optional(path: str, name: str) -> np.ndarray | None:
        return arrays.get((path, name))

    if any(path == "data/fields.npz" for path, _name in arrays):
        expected = (config.problem.phase_space_dofs,)
        for name in ("dof_index", "fom_angular_flux", "rom_angular_flux", "discrepancy"):
            array = optional("data/fields.npz", name)
            if array is None or array.shape != expected:
                raise ValueError(f"fields.npz {name} must have shape {expected}")
        field_time = optional("data/fields.npz", "field_time")
        if field_time is None or field_time.shape != (1,):
            raise ValueError("fields.npz field_time must have shape (1,)")
    if any(path == "data/error_history.npz" for path, _name in arrays):
        expected = (config.time.output_count,)
        for name in ("time", "instantaneous_normalized_mass_error"):
            array = optional("data/error_history.npz", name)
            if array is None or array.shape != expected:
                raise ValueError(f"error_history.npz {name} must have shape {expected}")
        boundary = optional("data/error_history.npz", "training_end_time")
        if boundary is None or boundary.shape != (1,):
            raise ValueError("error_history.npz training_end_time must have shape (1,)")
    if any(path == "data/pod_spectrum.npz" for path, _name in arrays):
        names = (
            "pod_eigenvalues",
            "retained_energy_fraction",
            "unresolved_energy_fraction",
            "basis_dimensions",
        )
        values = [optional("data/pod_spectrum.npz", name) for name in names]
        if any(value is None or value.ndim != 1 for value in values):
            raise ValueError("pod_spectrum.npz requires four one-dimensional arrays")
        lengths = {value.size for value in values if value is not None}
        minimum = int(case.latent_dimension or 0) + int(case.lifting_dimension or 0)
        if len(lengths) != 1 or next(iter(lengths)) < minimum:
            raise ValueError("POD spectrum arrays must have equal length covering retained dimensions")
    if any(path == "data/convergence_data.npz" for path, _name in arrays):
        dimensions = optional("data/convergence_data.npz", "model_dimension")
        convergence = optional(
            "data/convergence_data.npz", "relative_convergence_metric"
        )
        if (
            dimensions is None
            or convergence is None
            or dimensions.ndim != 1
            or convergence.shape != dimensions.shape
        ):
            raise ValueError(
                "convergence_data.npz requires equal one-dimensional model_dimension and relative_convergence_metric arrays"
            )
        for name in (
            "online_runtime_seconds",
            "online_speedup",
            "ode_function_evaluations",
        ):
            value = optional("data/convergence_data.npz", name)
            if value is not None and value.shape != dimensions.shape:
                raise ValueError(
                    f"convergence_data.npz {name} must match model_dimension"
                )


def validate_publication_artifact(
    root: str | Path,
    *,
    catalog: PublicationCatalog,
) -> dict[str, Any]:
    """Validate provenance, scientific metadata, and every recorded NPZ array."""
    root = Path(root)
    for filename in ("config.json", "case.json", "manifest.json", "metrics.json", "diagnostics.json"):
        if not (root / filename).is_file():
            raise ValueError(f"publication artifact is missing {filename}")
    case_data = _read_json(root / "case.json")
    case = catalog.get(case_data.get("case_id", ""))
    if case_data != case.to_dict():
        raise ValueError("case.json does not match the catalog case")
    manifest = _read_json(root / "manifest.json")
    config = OneDConfig.from_dict(_read_json(root / "config.json"))
    expected = {
        "case_id": case.case_id,
        "experiment_catalog_checksum": catalog.checksum(),
        "base_configuration_path": case.base_configuration_path,
        "base_configuration_checksum": case.base_configuration_checksum,
        "benchmark_variant": BENCHMARK_VARIANT,
        "manuscript_deviation": EXPECTED_DEVIATION,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"publication manifest has invalid {key}")
    if config.checksum() != case.base_configuration_checksum:
        resolved = resolve_case_configuration(case)
        if config.canonical_json() != resolved.canonical_json():
            raise ValueError("config.json is neither the base nor resolved case configuration")
    snapshot = manifest.get("input_snapshot", {})
    if snapshot.get("filename") != case.required_input_snapshot:
        raise ValueError("publication manifest has the wrong snapshot filename")
    if not isinstance(snapshot.get("path"), str) or not snapshot["path"]:
        raise ValueError("publication manifest requires an input snapshot path")
    checksum = snapshot.get("checksum_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError("input snapshot checksum must be a SHA-256 hex string")
    if snapshot.get("shape") != list(config.expected_snapshot_shape):
        raise ValueError("input snapshot shape does not match the resolved configuration")
    if not isinstance(snapshot.get("dtype"), str):
        raise ValueError("input snapshot dtype must be a string")
    try:
        snapshot_dtype = np.dtype(snapshot["dtype"])
    except TypeError:
        raise ValueError("input snapshot dtype is invalid") from None
    if snapshot_dtype.kind not in "fc":
        raise ValueError("input snapshot dtype must be floating or complex")
    model = manifest.get("model", {})
    if (
        model.get("type") != case.model_type
        or model.get("operator_construction") != case.operator_construction
        or model.get("N_r") != case.latent_dimension
        or model.get("N_q") != case.lifting_dimension
    ):
        raise ValueError("publication manifest model metadata does not match the case")
    if (
        model.get("inference_tolerance") != case.inference_tolerance
        or model.get("maximum_iterations") != case.maximum_iterations
    ):
        raise ValueError("publication manifest inference controls do not match the case")
    regularization = model.get("regularization", {})
    if regularization != {
        "gamma": case.lifting_regularization_gamma,
        "lambda_L": case.lambda_L,
        "lambda_Q": case.lambda_Q,
    }:
        raise ValueError("publication manifest regularization does not match the case")
    if manifest.get("training_interval") != list(case.training_interval):
        raise ValueError("publication manifest training interval does not match the case")
    if manifest.get("output_times") != {
        "initial": config.time.initial_time,
        "final": config.time.final_time,
        "spacing": config.time.output_spacing,
        "count": config.time.output_count,
    }:
        raise ValueError("publication manifest output-time metadata is invalid")
    if manifest.get("metric_definitions") != publication_metric_definitions():
        raise ValueError("publication manifest metric definitions are invalid")
    solver = manifest.get("solver", {})
    if not isinstance(solver.get("status"), str) or not solver["status"]:
        raise ValueError("publication manifest solver status is invalid")
    if solver.get("success") not in {None, True, False}:
        raise ValueError("publication manifest solver success must be null or boolean")
    if solver.get("message") is not None and not isinstance(solver["message"], str):
        raise ValueError("publication manifest solver message must be null or a string")
    validate_timing_metadata(manifest.get("timing", {}))
    arrays = _validate_array_records(root, manifest.get("arrays", []))
    _validate_case_array_shapes(arrays, case, config)
    return manifest


def execute_publication_case(
    catalog: PublicationCatalog,
    case: PublicationCase,
    *,
    input_snapshot: str | Path,
    execute: bool = False,
    output_root: str | Path = "results/1d/publication",
    run_directory: str | Path | None = None,
    run_id: str | None = None,
    shared_offline_directory: str | Path | None = None,
) -> PublicationRunDirectory:
    """Execute one fully specified case; imports scientific stages only after guards."""
    if not execute:
        raise PermissionError("publication execution requires execute=True")
    if not case.execution_allowed or not case.fully_specified:
        raise PublicationExecutionRefused(
            f"{case.case_id} requires author input: " + "; ".join(case.missing_information)
        )
    snapshot_path = Path(input_snapshot)
    base_config = resolve_base_configuration(case)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"required sigmoid benchmark snapshot is missing: {snapshot_path}")
    from .fom import inspect_snapshot

    inspection = inspect_snapshot(snapshot_path, base_config)
    if not inspection.compatible:
        raise ValueError(
            "input snapshot is incompatible with the legacy sigmoid configuration: "
            + "; ".join(inspection.compatibility_errors)
        )
    config = resolve_case_configuration(case)
    checksum = sha256_file(snapshot_path)
    run = create_publication_run_directory(
        catalog,
        case,
        config,
        input_snapshot=snapshot_path,
        input_snapshot_checksum=checksum,
        input_snapshot_shape=inspection.shape,
        input_snapshot_dtype=inspection.dtype,
        run_id=run_id,
        output_root=output_root,
        run_directory=run_directory,
    )
    shared = None
    if shared_offline_directory is not None:
        from .shared_offline import load_shared_offline_artifacts

        shared = load_shared_offline_artifacts(
            shared_offline_directory,
            base_config,
            dataset_sha256=checksum,
        )
        manifest = _read_json(run.manifest_path)
        manifest["shared_offline"] = {
            "path": str(shared.root),
            "manifest_sha256": sha256_file(shared.root / "manifest.json"),
            "dataset_sha256": checksum,
            "reused_derivatives": True,
            "reused_pod_svd": True,
        }
        _write_json(run.manifest_path, manifest)
    started = time.perf_counter()
    try:
        if case.model_type == "pod_analysis":
            if shared is None:
                _execute_pod_case(run, config, snapshot_path)
            else:
                _execute_pod_case_from_shared(run, config, shared)
        else:
            if shared is None:
                _execute_rom_case(run, case, config, snapshot_path)
            else:
                _execute_rom_case_from_shared(
                    run,
                    case,
                    config,
                    snapshot_path,
                    shared,
                )
        elapsed = time.perf_counter() - started
        timing = None
        if shared is None:
            timing = {
                "online_runtime_seconds": None,
                "offline_runtime_seconds": None,
                "total_runtime_seconds": elapsed,
                "speedup_basis": None,
                "included_stages": ["combined publication case pipeline"],
                "excluded_stages": [],
                "classification": "combined_total_only_not_publication_speedup",
            }
        update_publication_run(
            run,
            solver={"status": "completed", "success": True, "message": None},
            timing=timing,
        )
    except Exception as error:
        update_publication_run(
            run,
            solver={"status": "failed", "success": False, "message": str(error)},
        )
        raise
    return run


def _execute_pod_case(
    run: PublicationRunDirectory,
    config: OneDConfig,
    snapshot_path: Path,
) -> None:
    from .rom import compute_pod_data, initialize_rom_context, load_and_validate_snapshots

    snapshot = load_and_validate_snapshots(config, str(snapshot_path))
    context = initialize_rom_context(
        config,
        snapshot,
        model_name="linear",
        operator_choice="projected",
    )
    compute_pod_data(context)
    curves = pod_energy_curves(context.model.svd_val)
    write_npz_artifact(
        run,
        "pod_spectrum.npz",
        {
            "pod_eigenvalues": curves.eigenvalues,
            "retained_energy_fraction": curves.retained_energy_fraction,
            "unresolved_energy_fraction": curves.unresolved_energy_fraction,
            "basis_dimensions": curves.basis_dimensions,
        },
    )
    update_publication_run(
        run,
        metrics={
            "maximum_total_basis_dimension": 564,
            "highlighted_N_r": 16,
            "highlighted_N_q": 548,
        },
        diagnostics={
            "training_snapshot_count": int(context.training_indices.size),
            "steady_state_centering": True,
        },
    )


def _execute_pod_case_from_shared(
    run: PublicationRunDirectory,
    config: OneDConfig,
    shared: Any,
) -> None:
    started = time.perf_counter()
    curves = pod_energy_curves(shared.singular_values)
    write_npz_artifact(
        run,
        "pod_spectrum.npz",
        {
            "pod_eigenvalues": curves.eigenvalues,
            "retained_energy_fraction": curves.retained_energy_fraction,
            "unresolved_energy_fraction": curves.unresolved_energy_fraction,
            "basis_dimensions": curves.basis_dimensions,
            "highlighted_latent_dimension": np.asarray([16], dtype=int),
            "highlighted_lifting_dimension": np.asarray([548], dtype=int),
            "highlighted_total_dimension": np.asarray([564], dtype=int),
        },
    )
    elapsed = time.perf_counter() - started
    update_publication_run(
        run,
        metrics={
            "maximum_total_basis_dimension": 564,
            "highlighted_N_r": 16,
            "highlighted_N_q": 548,
            "retained_energy_at_N_r": float(curves.retained_energy_fraction[15]),
            "unresolved_energy_at_N_r": float(curves.unresolved_energy_fraction[15]),
            "retained_energy_at_total_dimension": float(
                curves.retained_energy_fraction[563]
            ),
            "unresolved_energy_at_total_dimension": float(
                curves.unresolved_energy_fraction[563]
            ),
        },
        diagnostics={
            "training_snapshot_count": int(shared.training_indices.size),
            "steady_state_centering": True,
            "shared_pod_reused": True,
        },
        timing={
            "online_runtime_seconds": None,
            "offline_runtime_seconds": elapsed,
            "total_runtime_seconds": elapsed,
            "speedup_basis": None,
            "included_stages": ["shared POD spectrum extraction and artifact write"],
            "excluded_stages": [
                "production FOM",
                "shared derivative computation",
                "shared full POD SVD",
            ],
            "classification": "artifact_extraction_only_not_publication_speedup",
        },
    )
def _execute_rom_case(
    run: PublicationRunDirectory,
    case: PublicationCase,
    config: OneDConfig,
    snapshot_path: Path,
) -> None:
    from .fom import build_time_array
    from .rom import partition_time_indices, run_selected_rom

    training_indices, _ = partition_time_indices(
        build_time_array(config), config.time.training_end_time
    )

    result = run_selected_rom(
        config,
        str(snapshot_path),
        model_name=case.model_type,
        operator_choice=str(case.operator_construction),
        regularization_scale=float(training_indices.size),
    )
    target_time = 2.5
    index = int(round((target_time - config.time.initial_time) / config.time.output_spacing))
    snapshot = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
    fom_field = np.asarray(snapshot[:, index])
    rom_field = np.asarray(result.reconstructed_state[:, index])
    write_npz_artifact(
        run,
        "fields.npz",
        {
            "field_time": np.asarray([target_time]),
            "dof_index": np.arange(fom_field.size, dtype=int),
            "fom_angular_flux": fom_field,
            "rom_angular_flux": rom_field,
            "discrepancy": fom_field - rom_field,
        },
    )
    write_npz_artifact(
        run,
        "error_history.npz",
        {
            "time": result.time,
            "instantaneous_normalized_mass_error": result.errors,
            "training_end_time": np.asarray([config.time.training_end_time]),
        },
    )
    update_publication_run(
        run,
        metrics={
            "instantaneous_error_history_metric_id": "instantaneous_steady_state_normalized_mass_error",
            "maximum_instantaneous_error": float(np.max(result.errors)),
            "mean_instantaneous_error_summary": float(np.mean(result.errors)),
            "publication_convergence_metric": None,
        },
        diagnostics=result.diagnostics,
        solver={
            "status": "rom_integrated",
            "success": bool(result.diagnostics["solver_success"]),
            "message": result.diagnostics["solver_message"],
        },
    )


def _execute_rom_case_from_shared(
    run: PublicationRunDirectory,
    case: PublicationCase,
    config: OneDConfig,
    snapshot_path: Path,
    shared: Any,
) -> None:
    from .rom import (
        construct_inferred_operators,
        construct_nonlinear_lifting,
        construct_projected_operators,
        initialize_rom_context_from_precomputed,
        integrate_selected_rom,
        reconstruct_and_compute_errors_chunked,
        regularization_diagnostics,
    )
    from Nonlinear_Manifold_ROM import ReducedIntegrationError

    snapshot = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
    context = initialize_rom_context_from_precomputed(
        config,
        snapshot,
        steady_state=shared.steady_state,
        time=shared.time,
        training_indices=shared.training_indices,
        extrapolation_indices=shared.extrapolation_indices,
        derivatives=shared.derivatives,
        basis=shared.basis,
        singular_values=shared.singular_values,
        coefficients=shared.coefficients,
        model_name=case.model_type,
        operator_choice=str(case.operator_construction),
    )
    case_started = time.perf_counter()
    regularization_scale = float(context.training_indices.size)
    regularization = regularization_diagnostics(
        context, regularization_scale=regularization_scale
    )
    lifting_started = time.perf_counter()
    construct_nonlinear_lifting(
        context, regularization_scale=regularization_scale
    )
    lifting_seconds = time.perf_counter() - lifting_started

    projection_started = time.perf_counter()
    construct_projected_operators(context)
    projection_seconds = time.perf_counter() - projection_started

    inference_seconds = 0.0
    if context.operator_choice == "inferred":
        inference_started = time.perf_counter()
        try:
            construct_inferred_operators(
                context, regularization_scale=regularization_scale
            )
        except Exception:
            inference_seconds = time.perf_counter() - inference_started
            update_publication_run(
                run,
                diagnostics={
                    "shared_offline_reused": True,
                    "inference": context.model.inference_diagnostics,
                    "inference_elapsed_seconds": inference_seconds,
                    "regularization": regularization,
                },
            )
            raise
        inference_seconds = time.perf_counter() - inference_started

    initial_started = time.perf_counter()
    context.model.compute_initial_conditions()
    initial_seconds = time.perf_counter() - initial_started
    initial = np.asarray(context.model.initial_condition)
    if context.model_name == "linear":
        initial_fit_residual = float(
            np.linalg.norm(context.model.pod_global_coeff[:, 0] - initial)
        )
    else:
        lifted_initial = np.concatenate(
            (
                initial,
                context.model.nonlinear_lift_matrix
                @ context.model.nonlinear_function(initial),
            )
        )
        initial_fit_residual = float(
            np.linalg.norm(context.model.pod_global_coeff[:, 0] - lifted_initial)
        )

    online_started = time.perf_counter()
    try:
        integration = integrate_selected_rom(context)
    except ReducedIntegrationError as error:
        online_seconds = time.perf_counter() - online_started
        update_publication_run(
            run,
            diagnostics={
                "shared_offline_reused": True,
                "training_snapshot_count": int(context.training_indices.size),
                "extrapolation_snapshot_count": int(
                    context.extrapolation_indices.size
                ),
                "reduced_initial_condition": {
                    "dimension": int(initial.size),
                    "euclidean_norm": float(np.linalg.norm(initial)),
                    "finite": bool(np.all(np.isfinite(initial))),
                    "POD_coefficient_fit_residual": initial_fit_residual,
                },
                "failed_integration": error.diagnostics,
                "regularization": regularization,
                "stage_timing_seconds": {
                    "nonlinear_lifting": lifting_seconds,
                    "operator_projection": projection_seconds,
                    "operator_inference": inference_seconds,
                    "initial_condition": initial_seconds,
                    "online_reduced_ODE_before_failure": online_seconds,
                },
            },
            timing={
                "online_runtime_seconds": online_seconds,
                "offline_runtime_seconds": (
                    lifting_seconds
                    + projection_seconds
                    + inference_seconds
                    + initial_seconds
                ),
                "total_runtime_seconds": time.perf_counter() - case_started,
                "speedup_basis": None,
                "included_stages": [
                    "case-specific lifting",
                    "operator projection",
                    "reduced initial condition",
                    "failed reduced ODE solve",
                ],
                "excluded_stages": [
                    "production FOM",
                    "shared derivative computation",
                    "shared full POD SVD",
                    "reconstruction after failed integration",
                ],
                "classification": "failed_case_diagnostics_not_publication_speedup",
            },
        )
        raise
    online_seconds = time.perf_counter() - online_started

    target_time = 2.5
    target_index = int(
        round((target_time - config.time.initial_time) / config.time.output_spacing)
    )
    reconstruction_started = time.perf_counter()
    errors, selected_fields = reconstruct_and_compute_errors_chunked(
        context,
        integration.y,
        selected_indices=(target_index,),
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    fom_field = np.asarray(snapshot[:, target_index])
    rom_field = selected_fields[target_index]
    write_npz_artifact(
        run,
        "fields.npz",
        {
            "field_time": np.asarray([target_time]),
            "field_time_index": np.asarray([target_index], dtype=int),
            "dof_index": np.arange(fom_field.size, dtype=int),
            "fom_angular_flux": fom_field,
            "rom_angular_flux": rom_field,
            "discrepancy": fom_field - rom_field,
        },
    )
    write_npz_artifact(
        run,
        "error_history.npz",
        {
            "time": context.time,
            "instantaneous_normalized_mass_error": errors,
            "field_time": np.asarray([target_time]),
            "training_end_time": np.asarray([config.time.training_end_time]),
        },
    )
    total_seconds = time.perf_counter() - case_started
    projected_nonlinear = context.model.projectedNonlinear
    active_linear = (
        context.model.projectedLinear
        if context.operator_choice == "projected"
        else context.model.inferredLinear
    )
    active_nonlinear = None
    if context.model_name != "linear":
        active_nonlinear = (
            projected_nonlinear
            if context.operator_choice == "projected"
            else context.model.inferredNonlinear
        )
    inference_diagnostics = context.model.inference_diagnostics
    if context.operator_choice == "inferred" and context.model_name == "linear":
        inference_diagnostics = {
            "converged": True,
            "iteration_count": 0,
            "final_convergence_measure": 0.0,
            "termination_reason": "direct_linear_solve",
        }
    training_errors = errors[context.training_indices]
    extrapolation_errors = errors[context.extrapolation_indices]
    diagnostics = {
        "shared_offline_reused": True,
        "full_reconstructed_trajectory_constructed": False,
        "reconstruction_chunk_size": 256,
        "training_snapshot_count": int(context.training_indices.size),
        "extrapolation_snapshot_count": int(context.extrapolation_indices.size),
        "reduced_initial_condition": {
            "dimension": int(initial.size),
            "euclidean_norm": float(np.linalg.norm(initial)),
            "finite": bool(np.all(np.isfinite(initial))),
            "POD_coefficient_fit_residual": initial_fit_residual,
        },
        "operator_dimensions": {
            "linear": list(np.asarray(active_linear).shape),
            "nonlinear": None
            if active_nonlinear is None
            else list(np.asarray(active_nonlinear).shape),
        },
        "solver": {
            "success": bool(integration.success),
            "message": str(integration.message),
            "returned_output_times": int(integration.t.size),
            "nfev": int(integration.nfev),
            "njev": int(integration.njev),
            "nlu": int(integration.nlu),
            "final_time": float(integration.t[-1]),
        },
        "inference": inference_diagnostics,
        "regularization": regularization,
        "stage_timing_seconds": {
            "nonlinear_lifting": lifting_seconds,
            "operator_projection": projection_seconds,
            "operator_inference": inference_seconds,
            "initial_condition": initial_seconds,
            "online_reduced_ODE": online_seconds,
            "chunked_reconstruction_and_error": reconstruction_seconds,
        },
    }
    metrics = {
        "instantaneous_error_history_metric_id": "instantaneous_steady_state_normalized_mass_error",
        "maximum_instantaneous_error": float(np.max(errors)),
        "mean_instantaneous_error_summary": float(np.mean(errors)),
        "maximum_training_error": float(np.max(training_errors)),
        "mean_training_error": float(np.mean(training_errors)),
        "maximum_extrapolation_error": float(np.max(extrapolation_errors)),
        "mean_extrapolation_error": float(np.mean(extrapolation_errors)),
        "publication_convergence_metric": None,
    }
    update_publication_run(
        run,
        metrics=metrics,
        diagnostics=diagnostics,
        solver={
            "status": "rom_integrated",
            "success": bool(integration.success),
            "message": str(integration.message),
        },
        timing={
            "online_runtime_seconds": online_seconds,
            "offline_runtime_seconds": (
                lifting_seconds
                + projection_seconds
                + inference_seconds
                + initial_seconds
            ),
            "total_runtime_seconds": total_seconds,
            "speedup_basis": None,
            "included_stages": [
                "case-specific lifting",
                "operator projection",
                "operator inference when selected",
                "reduced initial condition",
                "reduced ODE solve",
                "chunked reconstruction and instantaneous error",
            ],
            "excluded_stages": [
                "production FOM",
                "shared derivative computation",
                "shared full POD SVD",
            ],
            "classification": "measured_case_stages_not_publication_speedup",
        },
    )
def _artifact_arrays(root: Path, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    paths = sorted({record["artifact_path"] for record in manifest["arrays"]})
    for relative in paths:
        with np.load(root / relative, allow_pickle=False) as archive:
            for name in archive.files:
                arrays[name] = np.asarray(archive[name])
    return arrays


def build_figure_data_bundle(
    figure: str,
    artifact_roots: Iterable[str | Path],
    *,
    catalog: PublicationCatalog,
    output_directory: str | Path,
) -> FigureDataBundle:
    """Collect validated result arrays without importing or running solvers."""
    if figure not in FIGURE_EXPECTED_CASES:
        raise ValueError("figure must be one of Figure 1 through Figure 5")
    collected: dict[str, np.ndarray] = {}
    cases: list[str] = []
    sources: list[dict[str, Any]] = []
    for artifact_root in artifact_roots:
        root = Path(artifact_root)
        manifest = validate_publication_artifact(root, catalog=catalog)
        if manifest["figure"] != figure:
            continue
        case_id = manifest["case_id"]
        if case_id in cases:
            raise ValueError(
                f"multiple artifacts were supplied for {case_id}; select one run explicitly"
            )
        cases.append(case_id)
        sources.append(
            {
                "case_id": case_id,
                "run_id": manifest["run_id"],
                "artifact_root": str(root),
                "input_snapshot_checksum": manifest["input_snapshot"]["checksum_sha256"],
            }
        )
        for name, array in _artifact_arrays(root, manifest).items():
            collected[f"{case_id}__{name}"] = array
    if not cases:
        raise FileNotFoundError(f"no validated {figure} sigmoid-benchmark artifacts were provided")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"figure-data output directory already exists: {output}")
    output.mkdir(parents=True)
    data_path = output / "figure_data.npz"
    metadata_path = output / "figure_data.json"
    np.savez_compressed(data_path, **collected)
    expected = FIGURE_EXPECTED_CASES[figure]
    complete = expected.issubset(set(cases))
    if figure == "Figure 4":
        present_series = {
            f"{catalog.get(case_id).operator_construction}_{catalog.get(case_id).model_type}"
            for case_id in cases
        }
        expected_series = FIGURE4_EXPECTED_SERIES
    elif figure in {"Figure 2", "Figure 3"}:
        present_series = {catalog.get(case_id).model_type for case_id in cases}
        expected_series = {"linear", "elementwise", "tensorial"}
    else:
        present_series = set()
        expected_series = set()
    metadata = {
        "schema_version": "1.0.0",
        "figure": figure,
        "benchmark_variant": BENCHMARK_VARIANT,
        "manuscript_deviation": EXPECTED_DEVIATION,
        "catalog_checksum": catalog.checksum(),
        "case_ids": sorted(cases),
        "expected_case_ids": sorted(expected),
        "series_membership": sorted(present_series),
        "expected_series_membership": sorted(expected_series),
        "missing_series": sorted(expected_series.difference(present_series)),
        "case_set_complete": complete,
        "complete_publication_reproduction": False,
        "publication_reproduction_limitation": (
            "The preserved localized-sigmoid initial condition differs from the "
            "manuscript statement, and no original publication dataset checksum is available."
        ),
        "status": "complete_input_set" if complete else "partial_input_set",
        "sources": sources,
        "array_metadata": _array_metadata(
            data_path.name,
            collected,
        ),
    }
    _write_json(metadata_path, metadata)
    return FigureDataBundle(
        figure=figure,
        root=output,
        metadata_path=metadata_path,
        data_path=data_path,
        case_ids=tuple(sorted(cases)),
        complete=complete,
    )


def plot_figure_data_bundle(
    bundle_directory: str | Path,
    *,
    output_directory: str | Path,
) -> list[Path]:
    """Plot only prebuilt figure-ready arrays; never launch FOM or ROM work."""
    bundle = Path(bundle_directory)
    metadata = _read_json(bundle / "figure_data.json")
    if metadata.get("benchmark_variant") != BENCHMARK_VARIANT:
        raise ValueError("figure data is not marked as the legacy sigmoid benchmark")
    if metadata.get("manuscript_deviation") != EXPECTED_DEVIATION:
        raise ValueError("figure data lacks manuscript-deviation metadata")
    data_path = bundle / "figure_data.npz"
    if not data_path.is_file():
        raise FileNotFoundError(f"missing figure-ready arrays: {data_path}")
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"plot output directory already exists: {output}")
    output.mkdir(parents=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(data_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    figure = metadata["figure"]
    if figure == "Figure 1":
        paths = _plot_figure1(arrays, output, plt)
    elif figure in {"Figure 2", "Figure 3"}:
        paths = _plot_fields_and_errors(arrays, output, plt, figure)
    else:
        paths = _plot_convergence(arrays, output, plt, figure)
    caption = {
        "benchmark_variant": BENCHMARK_VARIANT,
        "manuscript_deviation": EXPECTED_DEVIATION,
        "complete_publication_reproduction": metadata["complete_publication_reproduction"],
        "note": "Generated only from validated result artifacts; no solver was run.",
    }
    _write_json(output / "plot_metadata.json", caption)
    return paths


def _plot_figure1(arrays: dict[str, np.ndarray], output: Path, plt: Any) -> list[Path]:
    dimensions = next(value for name, value in arrays.items() if name.endswith("__basis_dimensions"))
    unresolved = next(
        value for name, value in arrays.items() if name.endswith("__unresolved_energy_fraction")
    )
    fig, axis = plt.subplots()
    axis.semilogy(dimensions, unresolved)
    axis.axvline(16, color="tab:orange", linestyle="--", label="N_r=16")
    axis.axvline(564, color="tab:green", linestyle=":", label="N_r+N_q=564")
    axis.set(xlabel="Retained POD dimension", ylabel="Unresolved energy fraction")
    axis.set_title("Figure 1 data — legacy sigmoid benchmark")
    axis.legend()
    path = output / "figure1_unresolved_energy.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [path]


def _plot_fields_and_errors(
    arrays: dict[str, np.ndarray], output: Path, plt: Any, figure: str
) -> list[Path]:
    paths: list[Path] = []
    case_ids = sorted({name.split("__", 1)[0] for name in arrays})
    for case_id in case_ids:
        prefix = case_id + "__"
        if prefix + "rom_angular_flux" in arrays:
            fig, axis = plt.subplots()
            axis.plot(arrays[prefix + "dof_index"], arrays[prefix + "rom_angular_flux"], label="ROM")
            axis.plot(arrays[prefix + "dof_index"], arrays[prefix + "fom_angular_flux"], label="FOM", alpha=0.7)
            axis.set_title(f"{case_id} — legacy sigmoid benchmark")
            axis.set(xlabel="Phase-space DOF", ylabel="Angular flux")
            axis.legend()
            path = output / f"{case_id}_field.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
            fig, axis = plt.subplots()
            axis.plot(arrays[prefix + "dof_index"], arrays[prefix + "discrepancy"])
            axis.set_title(f"{case_id} discrepancy — legacy sigmoid benchmark")
            axis.set(xlabel="Phase-space DOF", ylabel="FOM − ROM angular flux")
            path = output / f"{case_id}_discrepancy.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
        if prefix + "instantaneous_normalized_mass_error" in arrays:
            fig, axis = plt.subplots()
            axis.semilogy(
                arrays[prefix + "time"],
                arrays[prefix + "instantaneous_normalized_mass_error"],
            )
            axis.axvline(7.5, color="black", linestyle="--", label="training end")
            axis.axvline(2.5, color="tab:red", linestyle=":", label="field time")
            axis.set_title(f"{figure}: {case_id} — legacy sigmoid benchmark")
            axis.set(xlabel="Time", ylabel="Instantaneous normalized M-error")
            axis.legend()
            path = output / f"{case_id}_error_history.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)
    if not paths:
        raise ValueError(f"figure data contains no plottable {figure} arrays")
    return paths


def _plot_convergence(
    arrays: dict[str, np.ndarray], output: Path, plt: Any, figure: str
) -> list[Path]:
    dimension_names = [name for name in arrays if name.endswith("__model_dimension")]
    metric_names = [name for name in arrays if name.endswith("__relative_convergence_metric")]
    if not dimension_names or not metric_names:
        raise ValueError(
            f"validated {figure} data lacks model_dimension or relative_convergence_metric arrays"
        )
    fig, axis = plt.subplots()
    case_ids = sorted({name.split("__", 1)[0] for name in dimension_names})
    for case_id in case_ids:
        x = arrays.get(case_id + "__model_dimension")
        y = arrays.get(case_id + "__relative_convergence_metric")
        if x is not None and y is not None:
            axis.loglog(x, y, marker="o", label=case_id)
    axis.set_title(f"{figure} data — legacy sigmoid benchmark")
    axis.set(xlabel="Model dimension", ylabel="Author-defined relative convergence metric")
    axis.legend()
    path = output / f"{figure.lower().replace(' ', '_')}_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths = [path]
    speedup_names = [name for name in arrays if name.endswith("__online_speedup")]
    if speedup_names:
        fig, axis = plt.subplots()
        for case_id in case_ids:
            x = arrays.get(case_id + "__model_dimension")
            y = arrays.get(case_id + "__online_speedup")
            if x is not None and y is not None:
                axis.semilogx(x, y, marker="o", label=case_id)
        axis.set_title(f"{figure} online speed-up — legacy sigmoid benchmark")
        axis.set(xlabel="Model dimension", ylabel="Online speed-up")
        axis.legend()
        speedup_path = output / f"{figure.lower().replace(' ', '_')}_online_speedup.png"
        fig.savefig(speedup_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(speedup_path)
    return paths
