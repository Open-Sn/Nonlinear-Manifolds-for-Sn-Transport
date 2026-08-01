"""Lossless shared Phase-5 dataset and offline-reduction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import scipy

from .config import OneDConfig
from .fom import build_time_array, inspect_snapshot
from .problem import assemble_operators, build_problem
from .publication_artifacts import sha256_file
from .publication_experiments import BENCHMARK_VARIANT, EXPECTED_DEVIATION
from .publication_metrics import pod_energy_curves
from .rom import compute_derivatives, compute_steady_state, initialize_rom_context


SHARED_SCHEMA_VERSION = "1.0.0"
SHARED_ARRAY_FILENAMES = {
    "steady_state": "steady_state.npy",
    "time": "time.npy",
    "training_indices": "training_indices.npy",
    "extrapolation_indices": "extrapolation_indices.npy",
    "snapshot_derivatives": "snapshot_derivatives.npy",
    "pod_singular_values": "pod_singular_values.npy",
    "pod_basis": "pod_basis.npy",
    "pod_coefficients": "pod_coefficients.npy",
}


@dataclass(frozen=True)
class SharedOfflineArrays:
    root: Path
    manifest: dict[str, Any]
    steady_state: np.ndarray
    time: np.ndarray
    training_indices: np.ndarray
    extrapolation_indices: np.ndarray
    derivatives: np.ndarray
    singular_values: np.ndarray
    basis: np.ndarray
    coefficients: np.ndarray


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=True,
                timeout=10,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _runtime_metadata() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "platform": platform.platform(),
    }


def _array_record(path: Path, root: Path) -> dict[str, Any]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {
        "path": str(path.relative_to(root)),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "file_size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def create_dataset_summary(
    config: OneDConfig,
    snapshot_path: str | Path,
    output_path: str | Path,
    *,
    dataset_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate and summarize selected production states without copying the dataset."""
    snapshot_path = Path(snapshot_path)
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite dataset summary: {output_path}")
    inspection = inspect_snapshot(snapshot_path, config, include_sha256=False)
    if not inspection.compatible:
        raise ValueError("snapshot is incompatible: " + "; ".join(inspection.compatibility_errors))
    snapshot = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
    problem = build_problem(config)
    operators = assemble_operators(problem)
    time_values = (0.0, 0.1, 1.0, 2.5, 7.5, 10.0)
    indices = [
        int(round((value - config.time.initial_time) / config.time.output_spacing))
        for value in time_values
    ]
    spatial_dofs = config.problem.spatial_dofs
    angular_indices = np.unique(
        np.clip(
            np.asarray(
                [
                    0,
                    1,
                    spatial_dofs - 2,
                    spatial_dofs - 1,
                    spatial_dofs,
                    2 * spatial_dofs - 1,
                    2 * spatial_dofs,
                    3 * spatial_dofs - 1,
                    3 * spatial_dofs,
                    config.problem.phase_space_dofs - 2,
                    config.problem.phase_space_dofs - 1,
                ]
            ),
            0,
            config.problem.phase_space_dofs - 1,
        )
    )
    spatial_indices = np.unique(
        np.clip(
            np.asarray(
                [
                    0,
                    1,
                    spatial_dofs // 3 - 1,
                    spatial_dofs // 3,
                    2 * spatial_dofs // 3 - 1,
                    2 * spatial_dofs // 3,
                    spatial_dofs - 2,
                    spatial_dofs - 1,
                ]
            ),
            0,
            spatial_dofs - 1,
        )
    )
    selected: list[dict[str, Any]] = []
    for requested_time, index in zip(time_values, indices):
        state = np.asarray(snapshot[:, index])
        directions = state.reshape(
            config.problem.angular_ordinates, config.problem.spatial_dofs
        )
        scalar_flux = problem.quadrature.w_q @ directions
        squared_mass_norm = float(state @ operators.mass.dot(state))
        scale = max(1.0, float(np.max(np.abs(state))))
        negative = state[state < 0.0]
        selected.append(
            {
                "requested_time": requested_time,
                "index": index,
                "actual_time": float(build_time_array(config)[index]),
                "euclidean_norm": float(np.linalg.norm(state)),
                "mass_norm": float(np.sqrt(max(squared_mass_norm, 0.0))),
                "minimum": float(np.min(state)),
                "maximum": float(np.max(state)),
                "finite": bool(np.all(np.isfinite(state))),
                "negative_entry_count": int(negative.size),
                "minimum_relative_to_state_scale": float(np.min(state) / scale),
                "angular_flux_indices": angular_indices.tolist(),
                "angular_flux_values": state[angular_indices].tolist(),
                "scalar_flux_spatial_indices": spatial_indices.tolist(),
                "scalar_flux_values": scalar_flux[spatial_indices].tolist(),
            }
        )
    summary = {
        "schema_version": SHARED_SCHEMA_VERSION,
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_variant": BENCHMARK_VARIANT,
        "manuscript_deviation": EXPECTED_DEVIATION,
        "configuration_checksum_sha256": config.checksum(),
        "snapshot": {
            **inspection.to_dict(),
            "sha256": dataset_sha256 or sha256_file(snapshot_path),
            "canonical_filename_confirmed": snapshot_path.name
            == config.output.snapshot_filename,
        },
        "time_grid": {
            "initial": config.time.initial_time,
            "final": config.time.final_time,
            "spacing": config.time.output_spacing,
            "count": config.time.output_count,
        },
        "selected_states": selected,
        "corruption_indicators": {
            "nonfinite_count": 0,
            "shape_mismatch": False,
            "time_grid_mismatch": False,
            "note": (
                "Small negative entries are recorded by scale and are not automatically "
                "classified as physical defects."
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(output_path, summary)
    return summary


def build_shared_offline_artifacts(
    config: OneDConfig,
    snapshot_path: str | Path,
    output_directory: str | Path,
    *,
    dataset_sha256: str,
    retained_dimension: int = 564,
) -> Path:
    """Compute derivatives and the historical full SVD exactly once."""
    snapshot_path = Path(snapshot_path)
    root = Path(output_directory)
    if root.exists():
        raise FileExistsError(f"shared offline directory already exists: {root}")
    if retained_dimension < 1 or retained_dimension > min(config.expected_snapshot_shape):
        raise ValueError("retained_dimension is outside the snapshot matrix dimensions")
    inspection = inspect_snapshot(snapshot_path, config, include_sha256=False)
    if not inspection.compatible:
        raise ValueError("snapshot is incompatible: " + "; ".join(inspection.compatibility_errors))
    if sha256_file(snapshot_path) != dataset_sha256:
        raise ValueError("dataset checksum changed before shared offline construction")
    data = root / "data"
    metrics = root / "metrics"
    logs = root / "logs"
    data.mkdir(parents=True, exist_ok=False)
    metrics.mkdir(exist_ok=False)
    logs.mkdir(exist_ok=False)
    manifest_path = root / "manifest.json"
    started_utc = datetime.now(timezone.utc)
    started = time.perf_counter()
    manifest: dict[str, Any] = {
        "schema_version": SHARED_SCHEMA_VERSION,
        "status": "running",
        "start_time_utc": started_utc.isoformat(),
        "finish_time_utc": None,
        "elapsed_seconds": None,
        "benchmark_variant": BENCHMARK_VARIANT,
        "manuscript_deviation": EXPECTED_DEVIATION,
        "dataset": {
            "path": str(snapshot_path),
            "filename": snapshot_path.name,
            "sha256": dataset_sha256,
            "shape": list(config.expected_snapshot_shape),
            "dtype": inspection.dtype,
            "duplicated_in_shared_artifacts": False,
        },
        "configuration_checksum_sha256": config.checksum(),
        "git": _git_metadata(),
        "runtime": _runtime_metadata(),
        "retained_dimension": retained_dimension,
        "arrays": {},
        "timing_seconds": {},
        "diagnostics": {},
    }
    _write_json(manifest_path, manifest)
    try:
        snapshot = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
        setup_started = time.perf_counter()
        context = initialize_rom_context(
            config,
            snapshot,
            model_name="linear",
            operator_choice="projected",
        )
        setup_seconds = time.perf_counter() - setup_started
        residual = context.operators.system.dot(context.steady_state) - context.operators.boundary_source
        steady_residual_absolute = float(np.linalg.norm(residual))
        steady_residual_relative = steady_residual_absolute / max(
            float(np.linalg.norm(context.operators.boundary_source)), np.finfo(float).eps
        )

        derivative_started = time.perf_counter()
        compute_derivatives(context)
        derivative_seconds = time.perf_counter() - derivative_started

        pod_started = time.perf_counter()
        context.model.compute_pod(
            size_R=config.rom.latent_dimension,
            size_Q=retained_dimension - config.rom.latent_dimension,
        )
        pod_seconds = time.perf_counter() - pod_started

        arrays = {
            "steady_state": context.steady_state,
            "time": context.time,
            "training_indices": context.training_indices,
            "extrapolation_indices": context.extrapolation_indices,
            "snapshot_derivatives": context.model.global_derivative_set,
            "pod_singular_values": context.model.svd_val,
            "pod_basis": context.model.basis[:, :retained_dimension],
            "pod_coefficients": context.model.coefficients[:retained_dimension, :],
        }
        write_started = time.perf_counter()
        for name, array in arrays.items():
            np.save(data / SHARED_ARRAY_FILENAMES[name], np.asarray(array), allow_pickle=False)
        records = {
            name: _array_record(data / filename, root)
            for name, filename in SHARED_ARRAY_FILENAMES.items()
        }
        write_seconds = time.perf_counter() - write_started

        retained_basis = np.asarray(arrays["pod_basis"])
        gram = retained_basis.T @ context.operators.mass.dot(retained_basis)
        orthogonality_error = gram - np.eye(retained_dimension)
        curves = pod_energy_curves(np.asarray(arrays["pod_singular_values"]))
        highlighted = sorted({config.rom.latent_dimension, retained_dimension})
        energy = {
            str(dimension): {
                "retained_energy_fraction": float(
                    curves.retained_energy_fraction[dimension - 1]
                ),
                "unresolved_energy_fraction": float(
                    curves.unresolved_energy_fraction[dimension - 1]
                ),
            }
            for dimension in highlighted
        }
        diagnostics = {
            "steady_state_residual_absolute": steady_residual_absolute,
            "steady_state_residual_relative": steady_residual_relative,
            "training_snapshot_count": int(context.training_indices.size),
            "extrapolation_snapshot_count": int(context.extrapolation_indices.size),
            "centered_training_snapshots": {
                "stored": False,
                "source_description": (
                    "input snapshot columns at training_indices minus steady_state[:, None]"
                ),
                "shape": [
                    config.problem.phase_space_dofs,
                    int(context.training_indices.size),
                ],
                "dtype": str(context.model.global_derivative_set.dtype),
            },
            "pod_algorithm": (
                "numpy.linalg.svd(M_sqrt @ centered_training, full_matrices=False, "
                "compute_uv=True, hermitian=False)"
            ),
            "mass_orthonormality": {
                "expression": "U.T @ M @ U - I",
                "frobenius_norm": float(np.linalg.norm(orthogonality_error)),
                "maximum_absolute_entry": float(np.max(np.abs(orthogonality_error))),
            },
            "energy_by_dimension": energy,
        }
        finished = datetime.now(timezone.utc)
        manifest.update(
            {
                "status": "completed",
                "finish_time_utc": finished.isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "arrays": records,
                "timing_seconds": {
                    "setup_and_steady_state": setup_seconds,
                    "snapshot_derivatives": derivative_seconds,
                    "pod_svd": pod_seconds,
                    "artifact_writes_and_hashes": write_seconds,
                },
                "diagnostics": diagnostics,
            }
        )
        _write_json(metrics / "offline_metrics.json", diagnostics)
        _write_json(manifest_path, manifest)
        return root
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        _write_json(manifest_path, manifest)
        raise


def load_shared_offline_artifacts(
    root: str | Path,
    config: OneDConfig,
    *,
    dataset_sha256: str,
) -> SharedOfflineArrays:
    """Validate shared provenance and load every large array read-only."""
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"shared offline manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("shared offline artifact is not complete")
    if manifest.get("benchmark_variant") != BENCHMARK_VARIANT:
        raise ValueError("shared offline artifact has the wrong benchmark variant")
    if manifest.get("manuscript_deviation") != EXPECTED_DEVIATION:
        raise ValueError("shared offline artifact lacks deviation provenance")
    if manifest.get("configuration_checksum_sha256") != config.checksum():
        raise ValueError("shared offline configuration checksum mismatch")
    if manifest.get("dataset", {}).get("sha256") != dataset_sha256:
        raise ValueError("shared offline dataset checksum mismatch")
    loaded: dict[str, np.ndarray] = {}
    records = manifest.get("arrays", {})
    if set(records) != set(SHARED_ARRAY_FILENAMES):
        raise ValueError("shared offline manifest has incomplete array records")
    for name, filename in SHARED_ARRAY_FILENAMES.items():
        record = records[name]
        path = root / record["path"]
        if path.name != filename or not path.is_file():
            raise ValueError(f"shared offline array is missing: {name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != record["shape"] or str(array.dtype) != record["dtype"]:
            raise ValueError(f"shared offline array metadata mismatch: {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"shared offline array checksum mismatch: {name}")
        loaded[name] = array
    return SharedOfflineArrays(
        root=root,
        manifest=manifest,
        steady_state=loaded["steady_state"],
        time=loaded["time"],
        training_indices=loaded["training_indices"],
        extrapolation_indices=loaded["extrapolation_indices"],
        derivatives=loaded["snapshot_derivatives"],
        singular_values=loaded["pod_singular_values"],
        basis=loaded["pod_basis"],
        coefficients=loaded["pod_coefficients"],
    )
