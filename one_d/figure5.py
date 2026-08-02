"""Resumable execution and compact artifacts for sigmoid-benchmark Figure 5."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable

import numpy as np
import scipy

from .config import OneDConfig, load_config
from .problem import assemble_operators, build_problem
from .publication_artifacts import sha256_file
from .publication_experiments import (
    AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
    BENCHMARK_VARIANT,
    DEFAULT_CATALOG_PATH,
    LEGACY_CONFIG_CHECKSUM,
    PublicationCatalog,
    load_publication_catalog,
)
from .publication_metrics import (
    ONLINE_TIMING_DEFINITION,
    RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
    relative_space_time_l2_error_from_energies,
)
from .shared_offline import SharedOfflineArrays, load_shared_offline_artifacts


NQ_VALUES = (1, 2, 4, 8, 16, 32, 64, 128)
FIXED_RANK = 32
TRAINING_COUNT = 7501
EXPECTED_DATASET_CHECKSUM = (
    "a3885dc5a071f67afb514e3d130d15cd993737a174313084f7e1ed0911cef6b3"
)
FIGURE5_AGGREGATE_CASES = (
    "fig5_projected_nq_sweep",
    "fig5_inferred_nq_sweep",
)
SERIES = (
    "fixed_linear",
    "elementwise",
    "tensorial",
    "enlarged_linear",
    "best_projection",
)
INTEGRATED_SERIES = SERIES[:-1]
RESULT_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Figure5Case:
    case_id: str
    execution_kind: str
    model: str
    operators: str | None
    N_r: int | None
    N_q: int | None
    reduced_dynamical_dimension: int | None
    projection_dimension: int | None
    gamma: float | None
    lambda_L: float | None
    lambda_Q: float | None
    applied_lifting_ridge: float | None
    applied_linear_ridge: float | None
    applied_quadratic_ridge: float | None
    panel_membership: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["panel_membership"] = list(self.panel_membership)
        return value


@dataclass(frozen=True)
class Figure5ExecutionContext:
    config: OneDConfig
    catalog: PublicationCatalog
    snapshot_path: Path
    snapshot: np.ndarray
    shared: SharedOfflineArrays
    problem: Any
    operators: Any
    fom_manifest_path: Path
    fom_manifest: dict[str, Any]
    fom_integration_elapsed_seconds: float
    reference_energy: np.ndarray
    centered_energy: np.ndarray
    max_projection_coefficients: np.ndarray
    environment: dict[str, Any]


class Figure5CaseExecutionError(RuntimeError):
    """A failed Figure 5 stage with the available scientific diagnostics."""

    def __init__(self, message: str, diagnostics: dict[str, Any]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _write_json(path: Path, content: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _canonical_checksum(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}


def _hardware_metadata() -> dict[str, Any]:
    blas: dict[str, Any] = {}
    try:
        configuration = np.show_config(mode="dicts")
        blas = configuration.get("Build Dependencies", {}).get("blas", {})
    except (AttributeError, TypeError):
        blas = {"name": "unavailable"}
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas": {
            key: value
            for key, value in blas.items()
            if key in {"name", "version", "configuration", "lib directory"}
        },
    }


def figure5_cases() -> tuple[Figure5Case, ...]:
    """Expand the 50 unique solves and 8 shared projection evaluations."""
    cases: list[Figure5Case] = []
    for operators in ("projected", "inferred"):
        inferred = operators == "inferred"
        cases.append(
            Figure5Case(
                case_id=f"fig5_{operators}_fixed_linear_nr32",
                execution_kind="rom_integration",
                model="linear",
                operators=operators,
                N_r=FIXED_RANK,
                N_q=None,
                reduced_dynamical_dimension=FIXED_RANK,
                projection_dimension=None,
                gamma=None,
                lambda_L=0.0 if inferred else None,
                lambda_Q=None,
                applied_lifting_ridge=None,
                applied_linear_ridge=0.0 if inferred else None,
                applied_quadratic_ridge=None,
                panel_membership=(operators,),
            )
        )
        for model, gamma, lambda_q in (
            ("elementwise", 8.0e-7, 8.0e-7),
            ("tensorial", 2.5e-8, 4.0e-7),
        ):
            for nq in NQ_VALUES:
                cases.append(
                    Figure5Case(
                        case_id=f"fig5_{operators}_{model}_nr32_nq{nq}",
                        execution_kind="rom_integration",
                        model=model,
                        operators=operators,
                        N_r=FIXED_RANK,
                        N_q=nq,
                        reduced_dynamical_dimension=FIXED_RANK,
                        projection_dimension=None,
                        gamma=gamma,
                        lambda_L=0.0 if inferred else None,
                        lambda_Q=lambda_q if inferred else None,
                        applied_lifting_ridge=gamma * TRAINING_COUNT,
                        applied_linear_ridge=0.0 if inferred else None,
                        applied_quadratic_ridge=(
                            lambda_q * TRAINING_COUNT if inferred else None
                        ),
                        panel_membership=(operators,),
                    )
                )
        for nq in NQ_VALUES:
            dimension = FIXED_RANK + nq
            cases.append(
                Figure5Case(
                    case_id=f"fig5_{operators}_enlarged_linear_nr{dimension}",
                    execution_kind="rom_integration",
                    model="linear",
                    operators=operators,
                    N_r=dimension,
                    N_q=None,
                    reduced_dynamical_dimension=dimension,
                    projection_dimension=None,
                    gamma=None,
                    lambda_L=0.0 if inferred else None,
                    lambda_Q=None,
                    applied_lifting_ridge=None,
                    applied_linear_ridge=0.0 if inferred else None,
                    applied_quadratic_ridge=None,
                    panel_membership=(operators,),
                )
            )
    for nq in NQ_VALUES:
        dimension = FIXED_RANK + nq
        cases.append(
            Figure5Case(
                case_id=f"fig5_best_m_projection_dim{dimension}",
                execution_kind="projection_benchmark",
                model="best_projection",
                operators=None,
                N_r=None,
                N_q=nq,
                reduced_dynamical_dimension=None,
                projection_dimension=dimension,
                gamma=None,
                lambda_L=None,
                lambda_Q=None,
                applied_lifting_ridge=None,
                applied_linear_ridge=None,
                applied_quadratic_ridge=None,
                panel_membership=("projected", "inferred"),
            )
        )
    identifiers = [case.case_id for case in cases]
    if len(identifiers) != 58 or len(set(identifiers)) != len(identifiers):
        raise AssertionError("Figure 5 plan must contain 58 unique computations")
    if sum(case.execution_kind == "rom_integration" for case in cases) != 50:
        raise AssertionError("Figure 5 plan must contain 50 unique ROM solves")
    return tuple(cases)


def validate_figure5_catalog(catalog: PublicationCatalog) -> None:
    for case_id in FIGURE5_AGGREGATE_CASES:
        case = catalog.get(case_id)
        if not case.fully_specified or not case.execution_allowed:
            raise ValueError(f"Figure 5 aggregate case is not execution-ready: {case_id}")
        if case.benchmark_variant != BENCHMARK_VARIANT:
            raise ValueError("Figure 5 catalog has incompatible benchmark provenance")
        if case.latent_dimension != FIXED_RANK:
            raise ValueError("Figure 5 catalog must use N_r=32")
        if tuple(case.parameter_sweep["lifting_dimensions"]) != NQ_VALUES:
            raise ValueError("Figure 5 catalog has incompatible N_q values")
        if case.provenance.get("metric_id") != "relative_space_time_l2_error_v1":
            raise ValueError("Figure 5 catalog has incompatible metric provenance")
        if case.provenance.get("online_timing_id") != "rom_solve_ivp_only_v1":
            raise ValueError("Figure 5 catalog has incompatible timing provenance")


def _result_status(case_root: Path, case: Figure5Case) -> tuple[str, str]:
    manifest_path = case_root / case.case_id / "manifest.json"
    if not manifest_path.is_file():
        return "not_reused", "pending"
    manifest = _read_json(manifest_path)
    if manifest.get("case_definition_checksum_sha256") != _canonical_checksum(
        case.to_dict()
    ):
        raise ValueError(f"existing Figure 5 case definition changed: {case.case_id}")
    status = manifest.get("status")
    if status == "completed":
        validate_figure5_case_result(case_root / case.case_id, case)
        return "reused_completed_result", "completed"
    if status == "failed":
        return "retained_failed_result", "failed"
    return "resume_interrupted_case", "pending"


def figure5_execution_plan(
    *,
    run_id: str,
    output_root: str | Path = "results/1d/publication",
    catalog: PublicationCatalog | None = None,
) -> dict[str, Any]:
    catalog = catalog or load_publication_catalog(DEFAULT_CATALOG_PATH)
    validate_figure5_catalog(catalog)
    output_root = Path(output_root)
    case_root = output_root / "figure5_cases" / run_id
    entries = []
    for case in figure5_cases():
        reuse, status = _result_status(case_root, case)
        entries.append(
            {
                **case.to_dict(),
                "expected_output": str(case_root / case.case_id),
                "reuse_status": reuse,
                "execution_status": status,
            }
        )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "benchmark_variant": BENCHMARK_VARIANT,
        "catalog_checksum_sha256": catalog.checksum(),
        "metric_definition": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
        "timing_definition": ONLINE_TIMING_DEFINITION,
        "training_snapshot_count": TRAINING_COUNT,
        "unique_rom_solve_count": 50,
        "unique_projection_benchmark_count": 8,
        "fixed_linear_solve_count_per_operator": 1,
        "maximum_linear_reduced_dimension": 160,
        "maximum_nonlinear_latent_dimension": 32,
        "maximum_lifting_dimension": 128,
        "large_array_reuse": {
            "production_snapshot": True,
            "snapshot_derivatives": True,
            "pod_basis": True,
            "pod_coefficients": True,
            "pod_singular_values": True,
            "full_reconstructed_trajectories_saved": False,
        },
        "forbidden_scope": {
            "figure4_execution": False,
            "fom_execution": False,
            "derivative_recomputation": False,
            "pod_svd_recomputation": False,
            "regularization_search": False,
        },
        "case_root": str(case_root),
        "cases": entries,
    }


def _validate_fom_manifest(
    manifest_path: Path,
    snapshot_path: Path,
) -> tuple[dict[str, Any], float]:
    manifest = _read_json(manifest_path)
    snapshot = manifest.get("snapshot", {})
    execution = manifest.get("execution", {})
    if snapshot.get("content_sha256") != EXPECTED_DATASET_CHECKSUM:
        raise ValueError("validated FOM manifest has the wrong dataset checksum")
    if snapshot.get("shape") != [6000, 10001] or snapshot.get("dtype") != "float64":
        raise ValueError("validated FOM manifest has incompatible snapshot metadata")
    if not execution.get("solver_success") or not execution.get(
        "diagnostics", {}
    ).get("final_time_confirmed"):
        raise ValueError("validated production FOM did not complete to t=10")
    elapsed = float(execution.get("elapsed_seconds"))
    if not np.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError("validated FOM integration elapsed time is invalid")
    if snapshot_path.name != snapshot.get("filename"):
        raise ValueError("production snapshot filename does not match FOM manifest")
    return manifest, elapsed


def _energy_history(
    states: np.ndarray,
    mass: Any,
    *,
    chunk_size: int = 256,
) -> np.ndarray:
    if states.ndim != 2:
        raise ValueError("energy history requires a state-by-time array")
    energies = np.empty(states.shape[1], dtype=float)
    for start in range(0, states.shape[1], chunk_size):
        stop = min(start + chunk_size, states.shape[1])
        block = np.asarray(states[:, start:stop])
        energies[start:stop] = np.sum(block * mass.dot(block), axis=0)
    if not np.all(np.isfinite(energies)) or np.any(energies < -1.0e-12):
        raise ValueError("state energy history is invalid")
    return np.maximum(energies, 0.0)


def _build_shared_metric_inputs(
    snapshot: np.ndarray,
    shared: SharedOfflineArrays,
    mass: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_energy = np.empty(snapshot.shape[1], dtype=float)
    centered_energy = np.empty(snapshot.shape[1], dtype=float)
    coefficients = np.empty((160, snapshot.shape[1]), dtype=float)
    basis = np.asarray(shared.basis[:, :160])
    steady = np.asarray(shared.steady_state)
    for start in range(0, snapshot.shape[1], 256):
        stop = min(start + 256, snapshot.shape[1])
        fom = np.asarray(snapshot[:, start:stop])
        centered = fom - steady[:, None]
        mass_fom = mass.dot(fom)
        mass_centered = mass.dot(centered)
        reference_energy[start:stop] = np.sum(fom * mass_fom, axis=0)
        centered_energy[start:stop] = np.sum(centered * mass_centered, axis=0)
        coefficients[:, start:stop] = basis.T @ mass_centered
    if not all(
        np.all(np.isfinite(value))
        for value in (reference_energy, centered_energy, coefficients)
    ):
        raise ValueError("shared metric inputs contain non-finite values")
    return reference_energy, centered_energy, coefficients


def _load_or_create_shared_metric_inputs(
    phase_root: Path,
    snapshot: np.ndarray,
    shared: SharedOfflineArrays,
    mass: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    path = phase_root / "shared_metric_inputs.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as archive:
            reference = np.asarray(archive["reference_energy"])
            centered = np.asarray(archive["centered_energy"])
            coefficients = np.asarray(archive["max_projection_coefficients"])
        if reference.shape != (10001,) or centered.shape != (10001,):
            raise ValueError("shared metric energy histories have incompatible shapes")
        if coefficients.shape != (160, 10001):
            raise ValueError("shared projection coefficients have incompatible shape")
        return reference, centered, coefficients, path
    reference, centered, coefficients = _build_shared_metric_inputs(
        snapshot,
        shared,
        mass,
    )
    np.savez_compressed(
        path,
        time=np.asarray(shared.time),
        reference_energy=reference,
        centered_energy=centered,
        max_projection_coefficients=coefficients,
    )
    return reference, centered, coefficients, path


def _config_for_case(base: OneDConfig, case: Figure5Case) -> OneDConfig:
    if case.execution_kind != "rom_integration":
        raise ValueError("projection benchmarks do not resolve to a ROM configuration")
    rom = replace(
        base.rom,
        latent_dimension=int(case.N_r),
        lifting_dimension=int(case.N_q or 0),
        embedding_type=case.model,
        streaming_operators=str(case.operators),
        lifting_regularization=float(case.gamma or 0.0),
        linear_inference_regularization=float(case.lambda_L or 0.0),
        quadratic_inference_regularization_linear=0.0,
        quadratic_inference_regularization_elementwise=(
            float(case.lambda_Q)
            if case.model == "elementwise" and case.lambda_Q is not None
            else base.rom.quadratic_inference_regularization_elementwise
        ),
        quadratic_inference_regularization_tensorial=(
            float(case.lambda_Q)
            if case.model == "tensorial" and case.lambda_Q is not None
            else base.rom.quadratic_inference_regularization_tensorial
        ),
        nonlinear_inference_tolerance=1.0e-6,
        nonlinear_inference_maximum_iterations=100000,
    )
    return replace(base, rom=rom)


def _operator_diagnostics(context: Any) -> dict[str, Any]:
    model = context.model
    if context.operator_choice != "inferred":
        return {}
    if context.model_name == "linear":
        return {
            "initialization": "closed_form_linear_inference_no_alternating_minimization",
            "converged": True,
            "iteration_count": 0,
            "final_convergence_measure": 0.0,
            "termination_reason": "direct_linear_solve",
            "regression_residual_norm": float(
                np.linalg.norm(
                    model.projectedDerivativeLinear
                    + model.inferredLinear @ model.pod_linear_coeff
                )
            ),
            "linear_operator_norm": float(np.linalg.norm(model.inferredLinear)),
            "quadratic_operator_norm": None,
        }
    residual = (
        model.projectedDerivativeLinear
        + (model.projectedAbsorptionLinear - model.projectedScatteringLinear)
        @ model.pod_linear_coeff
        + (model.projectedAbsorptionNonlinear - model.projectedScatteringNonlinear)
        @ model.pod_nonlinear_coeff
        + model.inferredStreamingLinear @ model.pod_linear_coeff
        + model.inferredStreamingNonlinear @ model.pod_nonlinear_coeff
    )
    return {
        "initialization": "zero_linear_and_quadratic_operators",
        **dict(model.inference_diagnostics),
        "regression_residual_norm": float(np.linalg.norm(residual)),
        "linear_streaming_operator_norm": float(
            np.linalg.norm(model.inferredStreamingLinear)
        ),
        "quadratic_streaming_operator_norm": float(
            np.linalg.norm(model.inferredStreamingNonlinear)
        ),
        "linear_operator_norm": float(np.linalg.norm(model.inferredLinear)),
        "quadratic_operator_norm": float(np.linalg.norm(model.inferredNonlinear)),
    }


def _rom_error_energy_chunked(
    context: Any,
    reduced_state: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    error_energy = np.empty(context.time.size, dtype=float)
    reconstruction_seconds = 0.0
    metric_seconds = 0.0
    for start in range(0, context.time.size, 256):
        stop = min(start + 256, context.time.size)
        reconstruction_started = time.perf_counter()
        reconstruction = context.model.reconstruct(reduced_state[:, start:stop])
        reconstruction_seconds += time.perf_counter() - reconstruction_started
        metric_started = time.perf_counter()
        difference = np.asarray(context.snapshot[:, start:stop]) - reconstruction
        error_energy[start:stop] = np.sum(
            difference * context.operators.mass.dot(difference),
            axis=0,
        )
        metric_seconds += time.perf_counter() - metric_started
    if not np.all(np.isfinite(error_energy)):
        raise RuntimeError("Figure 5 reconstruction produced non-finite error energy")
    return np.maximum(error_energy, 0.0), reconstruction_seconds, metric_seconds


def _case_manifest_base(
    case: Figure5Case,
    context: Figure5ExecutionContext,
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "running",
        "case_id": case.case_id,
        "case_definition": case.to_dict(),
        "case_definition_checksum_sha256": _canonical_checksum(case.to_dict()),
        "benchmark_variant": BENCHMARK_VARIANT,
        "initial_condition_provenance": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
        "metric_definition": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
        "timing_definition": ONLINE_TIMING_DEFINITION,
        "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
        "base_configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
        "catalog_checksum_sha256": context.catalog.checksum(),
        "shared_offline_manifest": {
            "path": str(context.shared.root / "manifest.json"),
            "sha256": sha256_file(context.shared.root / "manifest.json"),
        },
        "fom_manifest": {
            "path": str(context.fom_manifest_path),
            "sha256": sha256_file(context.fom_manifest_path),
            "integration_elapsed_seconds": context.fom_integration_elapsed_seconds,
        },
        "environment": context.environment,
        "source": _git_metadata(),
        "scientific_execution": {
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "figure4_case": False,
            "regularization_search": False,
        },
    }


def _execute_rom_case(
    case: Figure5Case,
    execution: Figure5ExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .rom import (
        construct_inferred_operators,
        construct_nonlinear_lifting,
        construct_projected_operators,
        initialize_rom_context_from_precomputed,
        integrate_selected_rom,
        regularization_diagnostics,
    )

    workflow_started = time.perf_counter()
    config = _config_for_case(execution.config, case)
    context_started = time.perf_counter()
    context = initialize_rom_context_from_precomputed(
        config,
        execution.snapshot,
        steady_state=execution.shared.steady_state,
        time=execution.shared.time,
        training_indices=execution.shared.training_indices,
        extrapolation_indices=execution.shared.extrapolation_indices,
        derivatives=execution.shared.derivatives,
        basis=execution.shared.basis,
        singular_values=execution.shared.singular_values,
        coefficients=execution.shared.coefficients,
        model_name=case.model,
        operator_choice=case.operators,
        problem=execution.problem,
        operators=execution.operators,
    )
    context_seconds = time.perf_counter() - context_started
    if context.training_indices.size != TRAINING_COUNT:
        raise ValueError("Figure 5 requires exactly 7,501 training snapshots")
    regularization = regularization_diagnostics(
        context,
        regularization_scale=float(TRAINING_COUNT),
    )
    lifting_seconds = 0.0
    projection_seconds = 0.0
    inference_seconds = 0.0
    initial_seconds = 0.0

    def failure_diagnostics(
        failed_stage: str,
        error: BaseException,
        *,
        inference: dict[str, Any] | None = None,
        solver: dict[str, Any] | None = None,
        initial: np.ndarray | None = None,
        initial_residual: float | None = None,
    ) -> dict[str, Any]:
        online = context.model.last_solve_ivp_elapsed_seconds
        return {
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "failed_stage": failed_stage,
            },
            "solver": solver
            or {
                "success": False,
                "message": "integration not attempted",
                "final_time": None,
                "returned_output_points": 0,
                "nfev": 0,
                "njev": 0,
                "nlu": 0,
            },
            "inference": inference or {},
            "regularization": {
                "training_snapshot_count": TRAINING_COUNT,
                "catalog_coefficients": {
                    "gamma": case.gamma,
                    "lambda_L": case.lambda_L,
                    "lambda_Q": case.lambda_Q,
                },
                "applied_gram_ridges": {
                    "gamma": case.applied_lifting_ridge,
                    "lambda_L": case.applied_linear_ridge,
                    "lambda_Q": case.applied_quadratic_ridge,
                },
                "scaling_count": 1,
                "implementation_diagnostics": regularization,
            },
            "reduced_initial_condition": {
                "dimension": int(initial.size) if initial is not None else None,
                "finite": bool(np.all(np.isfinite(initial))) if initial is not None else None,
                "euclidean_norm": float(np.linalg.norm(initial)) if initial is not None else None,
                "POD_coefficient_fit_residual": initial_residual,
            },
            "stage_timing": {
                "context_initialization_seconds": context_seconds,
                "lifting_construction_seconds": lifting_seconds,
                "projected_operator_construction_seconds": projection_seconds,
                "inference_seconds": inference_seconds,
                "initial_coordinate_fitting_seconds": initial_seconds,
                "online_integration_seconds": online,
                "reconstruction_seconds": 0.0,
                "metric_evaluation_seconds": 0.0,
                "total_case_workflow_seconds_before_artifact_write": (
                    time.perf_counter() - workflow_started
                ),
            },
            "full_reconstructed_trajectory_constructed": False,
            "finite": False,
        }
    expected_ridges = {
        "lifting_gram_ridge_actual": case.applied_lifting_ridge,
        "linear_inference_gram_ridge_actual": (
            float(case.applied_linear_ridge or 0.0)
        ),
        "quadratic_inference_gram_ridge_actual": (
            case.applied_quadratic_ridge
            if case.applied_quadratic_ridge is not None
            else regularization["quadratic_inference_gram_ridge_actual"]
        ),
    }
    if case.model != "linear" and not np.isclose(
        regularization["lifting_gram_ridge_actual"],
        expected_ridges["lifting_gram_ridge_actual"],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("lifting coefficient was not scaled exactly once")
    if case.operators == "inferred" and not np.isclose(
        regularization["linear_inference_gram_ridge_actual"],
        expected_ridges["linear_inference_gram_ridge_actual"],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("linear inference coefficient was not scaled exactly once")
    if case.model != "linear" and case.operators == "inferred" and not np.isclose(
        regularization["quadratic_inference_gram_ridge_actual"],
        expected_ridges["quadratic_inference_gram_ridge_actual"],
        rtol=0.0,
        atol=1.0e-15,
    ):
        raise ValueError("quadratic inference coefficient was not scaled exactly once")

    lifting_started = time.perf_counter()
    construct_nonlinear_lifting(
        context,
        regularization_scale=float(TRAINING_COUNT),
    )
    lifting_seconds = time.perf_counter() - lifting_started

    projection_started = time.perf_counter()
    construct_projected_operators(context)
    projection_seconds = time.perf_counter() - projection_started

    if case.operators == "inferred":
        inference_started = time.perf_counter()
        try:
            construct_inferred_operators(
                context,
                regularization_scale=float(TRAINING_COUNT),
            )
        except Exception as error:
            inference_seconds = time.perf_counter() - inference_started
            details = {
                "initialization": "zero_linear_and_quadratic_operators",
                **dict(getattr(context.model, "inference_diagnostics", {}) or {}),
            }
            for key, attribute in (
                ("linear_streaming_operator_norm", "inferredStreamingLinear"),
                ("quadratic_streaming_operator_norm", "inferredStreamingNonlinear"),
            ):
                value = getattr(context.model, attribute, None)
                details[key] = float(np.linalg.norm(value)) if value is not None else None
            raise Figure5CaseExecutionError(
                str(error),
                failure_diagnostics("operator_inference", error, inference=details),
            ) from error
        inference_seconds = time.perf_counter() - inference_started
    inference = _operator_diagnostics(context)
    if case.model != "linear" and case.operators == "inferred" and not inference.get(
        "converged"
    ):
        raise RuntimeError("nonlinear inference did not converge; integration refused")

    initial_started = time.perf_counter()
    context.model.compute_initial_conditions()
    initial_seconds = time.perf_counter() - initial_started
    initial = np.asarray(context.model.initial_condition)
    if case.model == "linear":
        initial_residual = float(
            np.linalg.norm(context.model.pod_global_coeff[:, 0] - initial)
        )
    else:
        lifted = np.concatenate(
            (
                initial,
                context.model.nonlinear_lift_matrix
                @ context.model.nonlinear_function(initial),
            )
        )
        initial_residual = float(
            np.linalg.norm(context.model.pod_global_coeff[:, 0] - lifted)
        )

    try:
        integration = integrate_selected_rom(context)
    except Exception as error:
        solver = dict(getattr(error, "diagnostics", {}) or {})
        if not solver:
            solver = {
                "success": False,
                "message": str(error),
                "final_time": None,
                "returned_output_points": 0,
                "nfev": 0,
                "njev": 0,
                "nlu": 0,
            }
        raise Figure5CaseExecutionError(
            str(error),
            failure_diagnostics(
                "online_integration",
                error,
                inference=inference,
                solver=solver,
                initial=initial,
                initial_residual=initial_residual,
            ),
        ) from error
    online_seconds = float(context.model.last_solve_ivp_elapsed_seconds)
    if not integration.success or float(integration.t[-1]) != 10.0:
        raise RuntimeError("Figure 5 reduced integration did not reach exactly t=10")

    error_energy, reconstruction_seconds, metric_energy_seconds = (
        _rom_error_energy_chunked(context, np.asarray(integration.y))
    )
    aggregate_started = time.perf_counter()
    aggregate_error = relative_space_time_l2_error_from_energies(
        error_energy,
        execution.reference_energy,
        context.time,
    )
    metric_seconds = metric_energy_seconds + time.perf_counter() - aggregate_started
    speedup = execution.fom_integration_elapsed_seconds / online_seconds
    stage_timing = {
        "context_initialization_seconds": context_seconds,
        "lifting_construction_seconds": lifting_seconds,
        "projected_operator_construction_seconds": projection_seconds,
        "inference_seconds": inference_seconds,
        "initial_coordinate_fitting_seconds": initial_seconds,
        "online_integration_seconds": online_seconds,
        "reconstruction_seconds": reconstruction_seconds,
        "metric_evaluation_seconds": metric_seconds,
        "total_case_workflow_seconds_before_artifact_write": (
            time.perf_counter() - workflow_started
        ),
    }
    metrics = {
        "metric_id": "relative_space_time_l2_error_v1",
        "relative_space_time_l2_error": aggregate_error,
        "online_timing_id": "rom_solve_ivp_only_v1",
        "fom_integration_elapsed_seconds": execution.fom_integration_elapsed_seconds,
        "rom_online_integration_elapsed_seconds": online_seconds,
        "online_speedup": speedup,
        "speedup_is_machine_specific": True,
    }
    diagnostics = {
        "solver": {
            "success": bool(integration.success),
            "message": str(integration.message),
            "nfev": int(integration.nfev),
            "njev": int(integration.njev),
            "nlu": int(integration.nlu),
            "final_time": float(integration.t[-1]),
            "returned_output_times": int(integration.t.size),
        },
        "inference": inference,
        "regularization": {
            "training_snapshot_count": TRAINING_COUNT,
            "catalog_coefficients": {
                "gamma": case.gamma,
                "lambda_L": case.lambda_L,
                "lambda_Q": case.lambda_Q,
            },
            "applied_gram_ridges": {
                "gamma": case.applied_lifting_ridge,
                "lambda_L": case.applied_linear_ridge,
                "lambda_Q": case.applied_quadratic_ridge,
            },
            "scaling_count": 1,
            "implementation_diagnostics": regularization,
        },
        "reduced_initial_condition": {
            "dimension": int(initial.size),
            "finite": bool(np.all(np.isfinite(initial))),
            "euclidean_norm": float(np.linalg.norm(initial)),
            "POD_coefficient_fit_residual": initial_residual,
        },
        "stage_timing": stage_timing,
        "full_reconstructed_trajectory_constructed": False,
        "reconstruction_chunk_size": 256,
        "finite": bool(
            np.isfinite(aggregate_error)
            and np.isfinite(online_seconds)
            and np.isfinite(speedup)
        ),
    }
    return metrics, diagnostics


def _execute_projection_case(
    case: Figure5Case,
    execution: Figure5ExecutionContext,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    dimension = int(case.projection_dimension)
    captured = np.sum(
        execution.max_projection_coefficients[:dimension, :] ** 2,
        axis=0,
    )
    error_energy = np.maximum(execution.centered_energy - captured, 0.0)
    metric_started = time.perf_counter()
    aggregate_error = relative_space_time_l2_error_from_energies(
        error_energy,
        execution.reference_energy,
        execution.shared.time,
    )
    metric_seconds = time.perf_counter() - metric_started
    metrics = {
        "metric_id": "relative_space_time_l2_error_v1",
        "relative_space_time_l2_error": aggregate_error,
        "online_timing_id": None,
        "fom_integration_elapsed_seconds": execution.fom_integration_elapsed_seconds,
        "rom_online_integration_elapsed_seconds": None,
        "online_speedup": None,
    }
    diagnostics = {
        "benchmark_type": "best_M_orthogonal_affine_projection",
        "projection_dimension": dimension,
        "steady_affine_component_restored": True,
        "mass_orthogonal_basis": True,
        "projection_speedup_applicable": False,
        "stage_timing": {
            "lifting_construction_seconds": 0.0,
            "projected_operator_construction_seconds": 0.0,
            "inference_seconds": 0.0,
            "initial_coordinate_fitting_seconds": 0.0,
            "online_integration_seconds": None,
            "reconstruction_seconds": 0.0,
            "metric_evaluation_seconds": metric_seconds,
            "total_case_workflow_seconds_before_artifact_write": (
                time.perf_counter() - started
            ),
        },
        "finite": bool(np.isfinite(aggregate_error)),
    }
    return metrics, diagnostics


def validate_figure5_case_result(root: Path, case: Figure5Case) -> dict[str, Any]:
    for filename in ("case.json", "manifest.json", "metrics.json", "diagnostics.json"):
        if not (root / filename).is_file():
            raise ValueError(f"Figure 5 case is missing {filename}: {case.case_id}")
    if _read_json(root / "case.json") != case.to_dict():
        raise ValueError(f"Figure 5 case definition mismatch: {case.case_id}")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError(f"Figure 5 case is not complete: {case.case_id}")
    if manifest.get("case_definition_checksum_sha256") != _canonical_checksum(
        case.to_dict()
    ):
        raise ValueError(f"Figure 5 case checksum mismatch: {case.case_id}")
    metrics = _read_json(root / "metrics.json")["metrics"]
    error = metrics.get("relative_space_time_l2_error")
    if not isinstance(error, (int, float)) or not np.isfinite(error) or error <= 0.0:
        raise ValueError(f"Figure 5 case error is not positive and finite: {case.case_id}")
    if case.execution_kind == "rom_integration":
        for field in (
            "rom_online_integration_elapsed_seconds",
            "online_speedup",
        ):
            value = metrics.get(field)
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"Figure 5 case has invalid {field}: {case.case_id}")
        diagnostics = _read_json(root / "diagnostics.json")["diagnostics"]
        if not diagnostics.get("solver", {}).get("success"):
            raise ValueError(f"Figure 5 ROM case solver failed: {case.case_id}")
        if diagnostics["solver"].get("final_time") != 10.0:
            raise ValueError(f"Figure 5 ROM case did not reach t=10: {case.case_id}")
    elif metrics.get("online_speedup") is not None:
        raise ValueError("projection benchmark must not have a speed-up")
    return manifest


def _execute_one_case(
    case: Figure5Case,
    case_root: Path,
    execution: Figure5ExecutionContext,
) -> str:
    root = case_root / case.case_id
    reuse, status = _result_status(case_root, case)
    if status == "completed":
        return reuse
    if status == "failed":
        return reuse
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "case.json", case.to_dict())
    manifest = _case_manifest_base(case, execution)
    manifest["start_time_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(root / "manifest.json", manifest)
    started = time.perf_counter()
    try:
        if case.execution_kind == "rom_integration":
            metrics, diagnostics = _execute_rom_case(case, execution)
        else:
            metrics, diagnostics = _execute_projection_case(case, execution)
        artifact_started = time.perf_counter()
        _write_json(root / "metrics.json", {"schema_version": "1.0.0", "metrics": metrics})
        _write_json(
            root / "diagnostics.json",
            {"schema_version": "1.0.0", "diagnostics": diagnostics},
        )
        artifact_seconds = time.perf_counter() - artifact_started
        diagnostics["stage_timing"]["artifact_writing_seconds"] = artifact_seconds
        diagnostics["stage_timing"]["total_case_workflow_seconds"] = (
            time.perf_counter() - started
        )
        _write_json(
            root / "diagnostics.json",
            {"schema_version": "1.0.0", "diagnostics": diagnostics},
        )
        manifest.update(
            {
                "status": "completed",
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "result_files": {
                    name: sha256_file(root / name)
                    for name in ("case.json", "metrics.json", "diagnostics.json")
                },
            }
        )
        _write_json(root / "manifest.json", manifest)
        validate_figure5_case_result(root, case)
        return "executed"
    except BaseException as error:
        interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
        manifest.update(
            {
                "status": "interrupted" if interrupted else "failed",
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json(root / "manifest.json", manifest)
        _write_json(root / "metrics.json", {"schema_version": "1.0.0", "metrics": {}})
        diagnostics = getattr(
            error,
            "diagnostics",
            {"failure": {"type": type(error).__name__, "message": str(error)}},
        )
        _write_json(
            root / "diagnostics.json",
            {"schema_version": "1.0.0", "diagnostics": diagnostics},
        )
        if interrupted:
            raise
        return "failed"


def prepare_figure5_execution_context(
    *,
    snapshot_path: str | Path,
    fom_manifest_path: str | Path,
    shared_offline_directory: str | Path,
    phase_root: Path,
    config_path: str | Path = "configs/1d/legacy_production.json",
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> tuple[Figure5ExecutionContext, dict[str, Any]]:
    config = load_config(config_path)
    if config.checksum() != LEGACY_CONFIG_CHECKSUM:
        raise ValueError("Figure 5 base configuration checksum changed")
    catalog = load_publication_catalog(catalog_path)
    validate_figure5_catalog(catalog)
    snapshot_path = Path(snapshot_path)
    if not snapshot_path.is_file():
        raise FileNotFoundError(f"production snapshot is missing: {snapshot_path}")
    actual_dataset_checksum = sha256_file(snapshot_path)
    if actual_dataset_checksum != EXPECTED_DATASET_CHECKSUM:
        raise ValueError("production snapshot checksum changed")
    fom_manifest_path = Path(fom_manifest_path)
    fom_manifest, fom_seconds = _validate_fom_manifest(
        fom_manifest_path,
        snapshot_path,
    )
    shared = load_shared_offline_artifacts(
        shared_offline_directory,
        config,
        dataset_sha256=actual_dataset_checksum,
    )
    if shared.training_indices.size != TRAINING_COUNT:
        raise ValueError("shared offline data must contain 7,501 training snapshots")
    snapshot = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
    problem = build_problem(config)
    operators = assemble_operators(problem)
    reference, centered, coefficients, shared_metric_path = (
        _load_or_create_shared_metric_inputs(
            phase_root,
            snapshot,
            shared,
            operators.mass,
        )
    )
    environment = _hardware_metadata()
    execution = Figure5ExecutionContext(
        config=config,
        catalog=catalog,
        snapshot_path=snapshot_path,
        snapshot=snapshot,
        shared=shared,
        problem=problem,
        operators=operators,
        fom_manifest_path=fom_manifest_path,
        fom_manifest=fom_manifest,
        fom_integration_elapsed_seconds=fom_seconds,
        reference_energy=reference,
        centered_energy=centered,
        max_projection_coefficients=coefficients,
        environment=environment,
    )
    assets = {
        "snapshot": {
            "path": str(snapshot_path),
            "sha256": actual_dataset_checksum,
            "shape": list(snapshot.shape),
            "dtype": str(snapshot.dtype),
        },
        "fom_manifest": {
            "path": str(fom_manifest_path),
            "sha256": sha256_file(fom_manifest_path),
            "integration_elapsed_seconds": fom_seconds,
        },
        "shared_offline_manifest": {
            "path": str(shared.root / "manifest.json"),
            "sha256": sha256_file(shared.root / "manifest.json"),
        },
        "shared_arrays": shared.manifest["arrays"],
        "shared_metric_inputs": {
            "path": str(shared_metric_path),
            "sha256": sha256_file(shared_metric_path),
            "large_scientific_arrays_recomputed": False,
        },
        "configuration_checksum_sha256": config.checksum(),
        "catalog_checksum_sha256": catalog.checksum(),
        "environment": environment,
    }
    return execution, assets


def execute_figure5_plan(
    *,
    run_id: str,
    snapshot_path: str | Path,
    fom_manifest_path: str | Path,
    shared_offline_directory: str | Path,
    output_root: str | Path = "results/1d/publication",
    config_path: str | Path = "configs/1d/legacy_production.json",
    catalog_path: str | Path = DEFAULT_CATALOG_PATH,
) -> dict[str, Any]:
    output_root = Path(output_root)
    phase_root = output_root / "phase7_runs" / run_id
    case_root = output_root / "figure5_cases" / run_id
    phase_root.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)
    plan_path = phase_root / "execution_plan.json"
    plan = figure5_execution_plan(
        run_id=run_id,
        output_root=output_root,
        catalog=load_publication_catalog(catalog_path),
    )
    if plan_path.is_file():
        existing = _read_json(plan_path)
        if existing.get("catalog_checksum_sha256") != plan["catalog_checksum_sha256"]:
            raise ValueError("cannot resume Figure 5 after a catalog change")
    plan["creation_or_resume_time_utc"] = datetime.now(timezone.utc).isoformat()
    plan["status"] = "running"
    _write_json(plan_path, plan)

    execution, assets = prepare_figure5_execution_context(
        snapshot_path=snapshot_path,
        fom_manifest_path=fom_manifest_path,
        shared_offline_directory=shared_offline_directory,
        phase_root=phase_root,
        config_path=config_path,
        catalog_path=catalog_path,
    )
    _write_json(phase_root / "asset_validation.json", assets)
    outcomes: dict[str, str] = {}
    for case in figure5_cases():
        outcomes[case.case_id] = _execute_one_case(case, case_root, execution)
        plan = figure5_execution_plan(
            run_id=run_id,
            output_root=output_root,
            catalog=execution.catalog,
        )
        plan["creation_or_resume_time_utc"] = datetime.now(timezone.utc).isoformat()
        plan["status"] = "running"
        plan["latest_outcomes"] = outcomes
        _write_json(plan_path, plan)
    final_plan = figure5_execution_plan(
        run_id=run_id,
        output_root=output_root,
        catalog=execution.catalog,
    )
    statuses = [entry["execution_status"] for entry in final_plan["cases"]]
    final_plan["creation_or_resume_time_utc"] = datetime.now(timezone.utc).isoformat()
    final_plan["status"] = "complete" if all(s == "completed" for s in statuses) else "partial"
    final_plan["latest_outcomes"] = outcomes
    final_plan["asset_validation"] = str(phase_root / "asset_validation.json")
    _write_json(plan_path, final_plan)
    return final_plan


def _case_result(case_root: Path, case: Figure5Case) -> dict[str, Any]:
    root = case_root / case.case_id
    validate_figure5_case_result(root, case)
    return {
        "case": case.to_dict(),
        "result_path": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "diagnostics_sha256": sha256_file(root / "diagnostics.json"),
        "metrics": _read_json(root / "metrics.json")["metrics"],
        "diagnostics": _read_json(root / "diagnostics.json")["diagnostics"],
    }


def _series_case(
    cases: Iterable[Figure5Case],
    operators: str,
    series: str,
    nq: int,
) -> Figure5Case:
    for case in cases:
        if series == "fixed_linear":
            match = (
                case.operators == operators
                and case.model == "linear"
                and case.N_r == FIXED_RANK
            )
        elif series == "enlarged_linear":
            match = (
                case.operators == operators
                and case.model == "linear"
                and case.N_r == FIXED_RANK + nq
            )
        elif series == "best_projection":
            match = case.model == "best_projection" and case.N_q == nq
        else:
            match = (
                case.operators == operators
                and case.model == series
                and case.N_q == nq
            )
        if match:
            return case
    raise KeyError(f"no Figure 5 case for {operators}/{series}/N_q={nq}")


def _array_metadata(arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    return [
        {"name": name, "shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    ]


def build_figure5_bundle(
    *,
    run_id: str,
    output_directory: str | Path,
    output_root: str | Path = "results/1d/publication",
    allow_partial: bool = False,
) -> Path:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Figure 5 bundle directory already exists: {output}")
    case_root = Path(output_root) / "figure5_cases" / run_id
    cases = figure5_cases()
    results: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for case in cases:
        try:
            results[case.case_id] = _case_result(case_root, case)
        except (FileNotFoundError, KeyError, ValueError):
            missing.append(case.case_id)
    complete = not missing
    if not complete and not allow_partial:
        raise ValueError("Figure 5 bundle requires every case; missing: " + ", ".join(missing))

    arrays: dict[str, np.ndarray] = {"n_q": np.asarray(NQ_VALUES, dtype=int)}
    points: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for operators in ("projected", "inferred"):
        points[operators] = {}
        for series in SERIES:
            entries: list[dict[str, Any]] = []
            error_values: list[float] = []
            speedup_values: list[float] = []
            online_values: list[float] = []
            for nq in NQ_VALUES:
                case = _series_case(cases, operators, series, nq)
                result = results.get(case.case_id)
                if result is None:
                    error_values.append(np.nan)
                    if series != "best_projection":
                        speedup_values.append(np.nan)
                        online_values.append(np.nan)
                    entries.append({"N_q": nq, "case_id": case.case_id, "status": "missing"})
                    continue
                metric = result["metrics"]
                error_values.append(float(metric["relative_space_time_l2_error"]))
                if series != "best_projection":
                    speedup_values.append(float(metric["online_speedup"]))
                    online_values.append(
                        float(metric["rom_online_integration_elapsed_seconds"])
                    )
                entries.append(
                    {
                        "N_q": nq,
                        "case_id": case.case_id,
                        "result_path": result["result_path"],
                        "manifest_sha256": result["manifest_sha256"],
                        "metrics_sha256": result["metrics_sha256"],
                        "diagnostics_sha256": result["diagnostics_sha256"],
                        "status": "complete",
                    }
                )
            prefix = f"{operators}__{series}"
            arrays[prefix + "__error"] = np.asarray(error_values, dtype=float)
            if series != "best_projection":
                arrays[prefix + "__online_speedup"] = np.asarray(
                    speedup_values,
                    dtype=float,
                )
                arrays[prefix + "__online_seconds"] = np.asarray(
                    online_values,
                    dtype=float,
                )
            points[operators][series] = entries
    output.mkdir(parents=True, exist_ok=False)
    npz_path = output / "figure5_data.npz"
    np.savez_compressed(npz_path, **arrays)
    phase_root = Path(output_root) / "phase7_runs" / run_id
    asset_validation = _read_json(phase_root / "asset_validation.json")
    metadata = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "figure": "Figure 5",
        "run_id": run_id,
        "benchmark_variant": BENCHMARK_VARIANT,
        "initial_condition_provenance": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
        "metric_definition": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
        "timing_definition": ONLINE_TIMING_DEFINITION,
        "N_r": FIXED_RANK,
        "N_q_values": list(NQ_VALUES),
        "case_set_status": "complete" if complete else "partial",
        "complete_publication_reproduction": False,
        "missing_case_ids": missing,
        "expected_unique_rom_solves": 50,
        "expected_projection_benchmarks": 8,
        "array_metadata": _array_metadata(arrays),
        "points": points,
        "case_records": results,
        "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
        "configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
        "catalog_checksum_sha256": asset_validation["catalog_checksum_sha256"],
        "fom_manifest": asset_validation["fom_manifest"],
        "shared_offline_manifest": asset_validation["shared_offline_manifest"],
        "environment": asset_validation["environment"],
        "scientific_execution": {
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "figure4_case": False,
            "regularization_search": False,
        },
        "data_file": {
            "path": str(npz_path),
            "sha256": sha256_file(npz_path),
        },
    }
    _write_json(output / "figure5_data.json", metadata)
    validate_figure5_bundle(output, require_complete=not allow_partial)
    return output


def validate_figure5_bundle(
    bundle_directory: str | Path,
    *,
    require_complete: bool = True,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = Path(bundle_directory)
    metadata_path = root / "figure5_data.json"
    npz_path = root / "figure5_data.npz"
    if not metadata_path.is_file() or not npz_path.is_file():
        raise ValueError("Figure 5 bundle requires figure5_data.json and figure5_data.npz")
    metadata = _read_json(metadata_path)
    if metadata.get("figure") != "Figure 5" or metadata.get(
        "benchmark_variant"
    ) != BENCHMARK_VARIANT:
        raise ValueError("Figure 5 bundle has incompatible identity metadata")
    if metadata.get("metric_definition") != RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION:
        raise ValueError("Figure 5 bundle has incompatible metric metadata")
    if metadata.get("timing_definition") != ONLINE_TIMING_DEFINITION:
        raise ValueError("Figure 5 bundle has incompatible timing metadata")
    if metadata.get("N_r") != FIXED_RANK or tuple(metadata.get("N_q_values", ())) != NQ_VALUES:
        raise ValueError("Figure 5 bundle has incompatible dimensions")
    if require_complete and metadata.get("case_set_status") != "complete":
        raise ValueError("Figure 5 bundle case set is partial")
    if sha256_file(npz_path) != metadata.get("data_file", {}).get("sha256"):
        raise ValueError("Figure 5 bundle data checksum mismatch")
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    records = {
        record["name"]: (tuple(record["shape"]), record["dtype"])
        for record in metadata.get("array_metadata", [])
    }
    if set(records) != set(arrays):
        raise ValueError("Figure 5 arrays do not match metadata")
    expected_names = {"n_q"}
    for operators in ("projected", "inferred"):
        for series in SERIES:
            expected_names.add(f"{operators}__{series}__error")
            if series != "best_projection":
                expected_names.add(f"{operators}__{series}__online_speedup")
                expected_names.add(f"{operators}__{series}__online_seconds")
    if set(arrays) != expected_names:
        raise ValueError("Figure 5 bundle is missing an expected series")
    if not np.array_equal(arrays["n_q"], np.asarray(NQ_VALUES)):
        raise ValueError("Figure 5 bundle N_q array changed")
    for name, array in arrays.items():
        shape, dtype = records[name]
        if array.shape != shape or str(array.dtype) != dtype:
            raise ValueError(f"Figure 5 array metadata mismatch: {name}")
        if name != "n_q" and require_complete:
            if array.shape != (8,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
                raise ValueError(f"Figure 5 series is not positive and finite: {name}")
    for operators in ("projected", "inferred"):
        fixed_error = arrays[f"{operators}__fixed_linear__error"]
        fixed_speed = arrays[f"{operators}__fixed_linear__online_speedup"]
        if require_complete and (
            not np.all(fixed_error == fixed_error[0])
            or not np.all(fixed_speed == fixed_speed[0])
        ):
            raise ValueError("fixed linear Figure 5 values must reuse one result")
        if not np.array_equal(
            arrays[f"{operators}__best_projection__error"],
            arrays["projected__best_projection__error"],
            equal_nan=True,
        ):
            raise ValueError("best-projection values must be shared by both panels")
    cases = figure5_cases()
    records = metadata.get("case_records", {})
    if not isinstance(records, dict):
        raise ValueError("Figure 5 case_records must be an object")
    if require_complete and set(records) != {case.case_id for case in cases}:
        raise ValueError("Figure 5 complete bundle must record every unique case")
    points = metadata.get("points", {})
    for operators in ("projected", "inferred"):
        for series in SERIES:
            entries = points.get(operators, {}).get(series)
            if not isinstance(entries, list) or len(entries) != len(NQ_VALUES):
                raise ValueError(f"Figure 5 point index is incomplete: {operators}/{series}")
            prefix = f"{operators}__{series}"
            for index, nq in enumerate(NQ_VALUES):
                case = _series_case(cases, operators, series, nq)
                entry = entries[index]
                if entry.get("N_q") != nq or entry.get("case_id") != case.case_id:
                    raise ValueError("Figure 5 point-to-case correspondence changed")
                record = records.get(case.case_id)
                if entry.get("status") == "missing":
                    if record is not None or require_complete:
                        raise ValueError("Figure 5 missing-point metadata is inconsistent")
                    continue
                if entry.get("status") != "complete" or not isinstance(record, dict):
                    raise ValueError("Figure 5 completed point lacks a case record")
                if record.get("case") != case.to_dict():
                    raise ValueError("Figure 5 case record definition changed")
                metrics = record.get("metrics", {})
                if float(metrics.get("relative_space_time_l2_error")) != float(
                    arrays[prefix + "__error"][index]
                ):
                    raise ValueError("Figure 5 error array does not match its case")
                if series == "best_projection":
                    if any(
                        metrics.get(field) is not None
                        for field in (
                            "online_timing_id",
                            "rom_online_integration_elapsed_seconds",
                            "online_speedup",
                        )
                    ):
                        raise ValueError("Figure 5 projection case has online timing")
                    continue
                online = float(metrics.get("rom_online_integration_elapsed_seconds"))
                speedup = float(metrics.get("online_speedup"))
                fom = float(metrics.get("fom_integration_elapsed_seconds"))
                if online != float(arrays[prefix + "__online_seconds"][index]):
                    raise ValueError("Figure 5 online-time array does not match its case")
                if speedup != float(arrays[prefix + "__online_speedup"][index]):
                    raise ValueError("Figure 5 speed-up array does not match its case")
                if speedup != fom / online:
                    raise ValueError("Figure 5 speed-up does not use the exact FOM timing")
    return metadata, arrays
