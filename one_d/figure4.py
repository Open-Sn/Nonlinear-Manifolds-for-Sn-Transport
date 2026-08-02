"""Reproducible regenerated-sigmoid search and execution for Figure 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Iterable

import numpy as np

from .figure5 import (
    EXPECTED_DATASET_CHECKSUM,
    Figure5Case,
    Figure5CaseExecutionError,
    Figure5ExecutionContext,
    TRAINING_COUNT,
    _config_for_case,
    _execute_rom_case,
    _operator_diagnostics,
    _rom_error_energy_chunked,
    prepare_figure5_execution_context,
)
from .publication_artifacts import sha256_file
from .publication_experiments import (
    AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
    BENCHMARK_VARIANT,
    DEFAULT_CATALOG_PATH,
    LEGACY_CONFIG_CHECKSUM,
    load_publication_catalog,
)
from .publication_metrics import (
    ONLINE_TIMING_DEFINITION,
    RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
    relative_space_time_l2_error_from_energies,
)


RANKS = (8, 16, 24, 32, 40, 48, 56, 64)
TOTAL_NONLINEAR_DIMENSION = 564
MODELS = ("elementwise", "tensorial")
OPERATORS = ("projected", "inferred")
GAMMA_GRID = tuple(float(value) for value in np.geomspace(7.0e-10, 5.0e-5, 9))
LAMBDA_Q_GRID = tuple(float(value) for value in np.geomspace(6.0e-9, 2.0e-4, 9))
TIE_RELATIVE_TOLERANCE = 1.0e-3
INFERENCE_TOLERANCE = 1.0e-6
MAXIMUM_INFERENCE_ITERATIONS = 100000
RESULT_LABEL = "regenerated_sigmoid_benchmark"
SELECTION_PROVENANCE = "regenerated_sigmoid_search"
SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class Figure4Candidate:
    candidate_id: str
    search_stage: str
    model: str
    operators: str
    N_r: int
    N_q: int
    gamma: float
    lambda_L: float | None
    lambda_Q: float | None
    applied_gamma_ridge: float
    applied_lambda_L_ridge: float | None
    applied_lambda_Q_ridge: float | None
    origin: str
    grid_index: int | None
    neighbor: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def rom_case(self) -> Figure5Case:
        return Figure5Case(
            case_id=self.candidate_id,
            execution_kind="rom_integration",
            model=self.model,
            operators=self.operators,
            N_r=self.N_r,
            N_q=self.N_q,
            reduced_dynamical_dimension=self.N_r,
            projection_dimension=None,
            gamma=self.gamma,
            lambda_L=self.lambda_L,
            lambda_Q=self.lambda_Q,
            applied_lifting_ridge=self.applied_gamma_ridge,
            applied_linear_ridge=self.applied_lambda_L_ridge,
            applied_quadratic_ridge=self.applied_lambda_Q_ridge,
            panel_membership=(self.operators,),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return repr(value)
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def canonical_checksum(value: Any) -> str:
    payload = json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_state_record(repository_root: str | Path = ".") -> dict[str, Any]:
    """Hash the complete non-result source state with length-prefixed framing."""
    root = Path(repository_root)
    tracked = subprocess.check_output(
        ["git", "ls-files"], cwd=root, text=True
    ).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root,
        text=True,
    ).splitlines()
    generated_parameter_copy = "configs/1d/publication/figure4_selected_parameters.json"
    paths = sorted(
        {
            path
            for path in tracked + untracked
            if not path.startswith("results/") and path != generated_parameter_copy
        }
    )
    digest = hashlib.sha256()
    for relative in paths:
        encoded_path = relative.encode("utf-8")
        content = (root / relative).read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {
        "algorithm": "sha256-length-prefixed-path-and-content-v1",
        "sha256": digest.hexdigest(),
        "file_count": len(paths),
        "git_commit": commit,
    }


def search_definition(
    *,
    run_id: str,
    source_state: dict[str, Any],
    catalog_checksum_sha256: str,
) -> dict[str, Any]:
    definition = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "figure4_regenerated_regularization_search_definition",
        "run_id": run_id,
        "result_label": RESULT_LABEL,
        "selection_provenance": SELECTION_PROVENANCE,
        "benchmark_variant": BENCHMARK_VARIANT,
        "source_state": source_state,
        "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
        "configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
        "catalog_checksum_sha256": catalog_checksum_sha256,
        "metric_id": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION["metric_id"],
        "online_timing_id": ONLINE_TIMING_DEFINITION["online_timing_id"],
        "dimensions": {
            "N_r": list(RANKS),
            "nonlinear_total_dimension": TOTAL_NONLINEAR_DIMENSION,
            "N_q": [TOTAL_NONLINEAR_DIMENSION - rank for rank in RANKS],
            "linear_N_q": None,
        },
        "candidate_grids": {
            "gamma": list(GAMMA_GRID),
            "lambda_Q": list(LAMBDA_Q_GRID),
            "lambda_L": 0.0,
            "grid_construction": "numpy.geomspace including both endpoints",
        },
        "regularization_scaling": {
            "semantics": "paper coefficient multiplied by training count exactly once",
            "training_snapshot_count": TRAINING_COUNT,
            "applied_gamma_ridge": "gamma * 7501",
            "applied_lambda_Q_ridge": "lambda_Q * 7501",
        },
        "selection": {
            "objective": "minimum relative_space_time_l2_error_v1",
            "gamma_source": "projected model search",
            "inferred_gamma_reuse": True,
            "tie_relative_tolerance": TIE_RELATIVE_TOLERANCE,
            "tie_rule": "among candidates within 0.1 percent of the minimum choose the larger coefficient",
            "refinement": "geometric midpoint to each available neighboring coarse value",
            "range_extension": False,
        },
        "admissibility": {
            "projected": [
                "lifting construction succeeds",
                "initial-coordinate fit succeeds",
                "integration succeeds exactly to t=10",
                "all outputs and approved metric are finite",
            ],
            "inferred": [
                "nonlinear inference converges below 1e-6 within 100000 iterations",
                "inferred operators are finite",
                "integration succeeds exactly to t=10",
                "all outputs and approved metric are finite",
            ],
        },
        "solver": {
            "method_and_tolerances": "unchanged from configs/1d/legacy_production.json",
            "inference_tolerance": INFERENCE_TOLERANCE,
            "maximum_inference_iterations": MAXIMUM_INFERENCE_ITERATIONS,
        },
        "planned_counts": {
            "projected_coarse": 144,
            "projected_refinement_maximum": 32,
            "inferred_coarse": 144,
            "inferred_refinement_maximum": 32,
            "nonlinear_candidate_maximum": 352,
            "linear_final_cases": 16,
            "nonlinear_final_cases": 32,
        },
        "scientific_execution": {
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "figure5_rerun": False,
            "search_outside_approved_ranges": False,
        },
    }
    return {**definition, "content_checksum_sha256": canonical_checksum(definition)}


def validate_search_definition(value: dict[str, Any]) -> None:
    payload = dict(value)
    checksum = payload.pop("content_checksum_sha256", None)
    if checksum != canonical_checksum(payload):
        raise ValueError("Figure 4 search-definition checksum mismatch")
    if tuple(value["candidate_grids"]["gamma"]) != GAMMA_GRID:
        raise ValueError("Figure 4 gamma grid changed")
    if tuple(value["candidate_grids"]["lambda_Q"]) != LAMBDA_Q_GRID:
        raise ValueError("Figure 4 lambda_Q grid changed")
    if value.get("result_label") != RESULT_LABEL:
        raise ValueError("Figure 4 result label changed")


def geometric_refinement_values(grid: Iterable[float], winner_index: int) -> list[tuple[str, float]]:
    values = tuple(float(value) for value in grid)
    if winner_index < 0 or winner_index >= len(values):
        raise IndexError("winner index is outside the coefficient grid")
    refined: list[tuple[str, float]] = []
    if winner_index > 0:
        refined.append(("lower", float(np.sqrt(values[winner_index - 1] * values[winner_index]))))
    if winner_index + 1 < len(values):
        refined.append(("upper", float(np.sqrt(values[winner_index] * values[winner_index + 1]))))
    return refined


def select_candidate(
    records: Iterable[dict[str, Any]],
    *,
    coefficient_field: str,
) -> dict[str, Any]:
    records = list(records)
    admissible = [record for record in records if record.get("admissible")]
    if not admissible:
        raise ValueError("no admissible regularization candidate")
    for record in admissible:
        error = record.get("relative_space_time_l2_error")
        coefficient = record["candidate"].get(coefficient_field)
        if not isinstance(error, (int, float)) or not np.isfinite(error) or error <= 0.0:
            raise ValueError("admissible candidate has an invalid objective")
        if not isinstance(coefficient, (int, float)) or coefficient <= 0.0:
            raise ValueError("candidate coefficient is invalid")
    minimum = min(float(record["relative_space_time_l2_error"]) for record in admissible)
    tied = [
        record
        for record in admissible
        if (float(record["relative_space_time_l2_error"]) - minimum) / minimum
        <= TIE_RELATIVE_TOLERANCE
    ]
    selected = max(tied, key=lambda record: float(record["candidate"][coefficient_field]))
    ordered = sorted(
        admissible,
        key=lambda record: (
            float(record["relative_space_time_l2_error"]),
            -float(record["candidate"][coefficient_field]),
        ),
    )
    return {
        "selected_candidate_id": selected["candidate"]["candidate_id"],
        "selected_coefficient": selected["candidate"][coefficient_field],
        "selected_error": selected["relative_space_time_l2_error"],
        "minimum_error": minimum,
        "tied_candidate_ids": [record["candidate"]["candidate_id"] for record in tied],
        "larger_regularization_tie_choice_applied": len(tied) > 1,
        "candidate_rank": 1 + ordered.index(selected),
        "admissible_count": len(admissible),
        "inadmissible_count": len(records) - len(admissible),
    }


def _coarse_candidate(
    *,
    model: str,
    operators: str,
    rank: int,
    coefficient: float,
    grid_index: int,
    gamma: float | None = None,
) -> Figure4Candidate:
    nq = TOTAL_NONLINEAR_DIMENSION - rank
    if operators == "projected":
        gamma_value = coefficient
        lambda_q = None
        stage = "projected_gamma_coarse"
        prefix = "gamma"
    else:
        if gamma is None:
            raise ValueError("inferred candidate requires selected gamma")
        gamma_value = gamma
        lambda_q = coefficient
        stage = "inferred_lambda_Q_coarse"
        prefix = "lambdaq"
    return Figure4Candidate(
        candidate_id=f"fig4_search_{prefix}_coarse_{model}_nr{rank}_{grid_index:02d}",
        search_stage=stage,
        model=model,
        operators=operators,
        N_r=rank,
        N_q=nq,
        gamma=gamma_value,
        lambda_L=0.0 if operators == "inferred" else None,
        lambda_Q=lambda_q,
        applied_gamma_ridge=gamma_value * TRAINING_COUNT,
        applied_lambda_L_ridge=0.0 if operators == "inferred" else None,
        applied_lambda_Q_ridge=(lambda_q * TRAINING_COUNT if lambda_q is not None else None),
        origin="coarse",
        grid_index=grid_index,
        neighbor=None,
    )


def _refined_candidate(
    *,
    model: str,
    operators: str,
    rank: int,
    coefficient: float,
    neighbor: str,
    gamma: float | None = None,
) -> Figure4Candidate:
    nq = TOTAL_NONLINEAR_DIMENSION - rank
    if operators == "projected":
        gamma_value = coefficient
        lambda_q = None
        prefix = "gamma"
        stage = "projected_gamma_refined"
    else:
        if gamma is None:
            raise ValueError("inferred refinement requires selected gamma")
        gamma_value = gamma
        lambda_q = coefficient
        prefix = "lambdaq"
        stage = "inferred_lambda_Q_refined"
    return Figure4Candidate(
        candidate_id=f"fig4_search_{prefix}_refined_{neighbor}_{model}_nr{rank}",
        search_stage=stage,
        model=model,
        operators=operators,
        N_r=rank,
        N_q=nq,
        gamma=gamma_value,
        lambda_L=0.0 if operators == "inferred" else None,
        lambda_Q=lambda_q,
        applied_gamma_ridge=gamma_value * TRAINING_COUNT,
        applied_lambda_L_ridge=0.0 if operators == "inferred" else None,
        applied_lambda_Q_ridge=(lambda_q * TRAINING_COUNT if lambda_q is not None else None),
        origin="refined",
        grid_index=None,
        neighbor=neighbor,
    )


def _candidate_record(root: Path, candidate: Figure4Candidate) -> dict[str, Any]:
    manifest = _read_json(root / "manifest.json")
    if manifest.get("candidate_definition_checksum_sha256") != canonical_checksum(
        candidate.to_dict()
    ):
        raise ValueError(f"candidate definition changed: {candidate.candidate_id}")
    metrics = _read_json(root / "metrics.json").get("metrics", {})
    return {
        "candidate": candidate.to_dict(),
        "artifact_path": str(root),
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "metrics_sha256": sha256_file(root / "metrics.json"),
        "diagnostics_sha256": sha256_file(root / "diagnostics.json"),
        "status": manifest.get("status"),
        "admissible": bool(manifest.get("admissible")),
        "relative_space_time_l2_error": metrics.get("relative_space_time_l2_error"),
        "online_speedup": metrics.get("online_speedup"),
        "online_seconds": metrics.get("rom_online_integration_elapsed_seconds"),
        "failure_reason": manifest.get("failure_reason"),
    }


def validate_candidate_result(root: Path, candidate: Figure4Candidate) -> dict[str, Any]:
    for filename in ("candidate.json", "manifest.json", "metrics.json", "diagnostics.json"):
        if not (root / filename).is_file():
            raise ValueError(f"candidate artifact is missing {filename}")
    if _read_json(root / "candidate.json") != candidate.to_dict():
        raise ValueError("candidate.json does not match its definition")
    record = _candidate_record(root, candidate)
    if record["status"] != "completed":
        raise ValueError("candidate artifact is not complete")
    if record["admissible"]:
        for field in ("relative_space_time_l2_error", "online_speedup", "online_seconds"):
            value = record[field]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"admissible candidate has invalid {field}")
        diagnostics = _read_json(root / "diagnostics.json")["diagnostics"]
        solver = diagnostics.get("solver", {})
        if not solver.get("success") or solver.get("final_time") != 10.0:
            raise ValueError("admissible candidate did not reach t=10")
        if candidate.operators == "inferred" and not diagnostics.get("inference", {}).get(
            "converged"
        ):
            raise ValueError("admissible inferred candidate did not converge")
    return record


def _execute_candidate(
    *,
    candidate: Figure4Candidate,
    root: Path,
    definition: dict[str, Any],
    execution: Figure5ExecutionContext,
    runner: Callable[[], tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        if manifest.get("candidate_definition_checksum_sha256") != canonical_checksum(
            candidate.to_dict()
        ):
            raise ValueError(f"cannot reuse changed candidate {candidate.candidate_id}")
        if manifest.get("status") == "completed":
            return validate_candidate_result(root, candidate)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "candidate.json", candidate.to_dict())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "admissible": False,
        "candidate_id": candidate.candidate_id,
        "candidate_definition_checksum_sha256": canonical_checksum(candidate.to_dict()),
        "search_definition_checksum_sha256": definition["content_checksum_sha256"],
        "result_label": RESULT_LABEL,
        "selection_provenance": SELECTION_PROVENANCE,
        "benchmark_variant": BENCHMARK_VARIANT,
        "initial_condition_provenance": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
        "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
        "configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
        "catalog_checksum_sha256": execution.catalog.checksum(),
        "metric_definition": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
        "timing_definition": ONLINE_TIMING_DEFINITION,
        "selected_status": "recorded_in_search_index_after_group_selection",
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_execution": {
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "figure5_rerun": False,
            "search_outside_approved_ranges": False,
        },
    }
    _write_json(manifest_path, manifest)
    failure_reason = None
    metrics: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    interrupted = False
    try:
        metrics, diagnostics = runner()
        admissible = bool(
            np.isfinite(metrics.get("relative_space_time_l2_error", np.nan))
            and np.isfinite(metrics.get("rom_online_integration_elapsed_seconds", np.nan))
            and diagnostics.get("solver", {}).get("success")
            and diagnostics.get("solver", {}).get("final_time") == 10.0
            and diagnostics.get("finite")
        )
        if candidate.operators == "inferred":
            admissible = admissible and bool(
                diagnostics.get("inference", {}).get("converged")
            )
        if not admissible:
            failure_reason = "candidate failed an admissibility condition"
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        raise
    except BaseException as error:
        admissible = False
        failure_reason = f"{type(error).__name__}: {error}"
        diagnostics = getattr(
            error,
            "diagnostics",
            {"failure": {"type": type(error).__name__, "message": str(error)}},
        )
    finally:
        if interrupted:
            manifest.update(
                {
                    "status": "interrupted",
                    "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                    "failure_reason": "KeyboardInterrupt or SystemExit",
                }
            )
        else:
            _write_json(root / "metrics.json", {"schema_version": SCHEMA_VERSION, "metrics": metrics})
            _write_json(
                root / "diagnostics.json",
                {"schema_version": SCHEMA_VERSION, "diagnostics": diagnostics},
            )
            manifest.update(
                {
                    "status": "completed",
                    "admissible": admissible,
                    "scientific_status": "admissible" if admissible else "inadmissible",
                    "failure_reason": failure_reason,
                    "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                    "result_files": {
                        filename: sha256_file(root / filename)
                        for filename in ("candidate.json", "metrics.json", "diagnostics.json")
                    },
                }
            )
        _write_json(manifest_path, manifest)
    return validate_candidate_result(root, candidate)


@dataclass
class InferenceCache:
    context: Any
    preparation: dict[str, Any]


def _prepare_inference_cache(
    candidate: Figure4Candidate,
    execution: Figure5ExecutionContext,
) -> InferenceCache:
    from .rom import (
        construct_nonlinear_lifting,
        construct_projected_operators,
        initialize_rom_context_from_precomputed,
    )

    config = _config_for_case(execution.config, candidate.rom_case())
    started = time.perf_counter()
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
        model_name=candidate.model,
        operator_choice="inferred",
        problem=execution.problem,
        operators=execution.operators,
    )
    context_seconds = time.perf_counter() - started
    lifting_started = time.perf_counter()
    construct_nonlinear_lifting(context, regularization_scale=float(TRAINING_COUNT))
    lifting_seconds = time.perf_counter() - lifting_started
    projection_started = time.perf_counter()
    construct_projected_operators(context)
    projection_seconds = time.perf_counter() - projection_started
    initial_started = time.perf_counter()
    context.model.compute_initial_conditions()
    initial_seconds = time.perf_counter() - initial_started
    initial = np.asarray(context.model.initial_condition)
    lifted = np.concatenate(
        (initial, context.model.nonlinear_lift_matrix @ context.model.nonlinear_function(initial))
    )
    residual = float(np.linalg.norm(context.model.pod_global_coeff[:, 0] - lifted))
    return InferenceCache(
        context=context,
        preparation={
            "cache_key": f"{candidate.model}_nr{candidate.N_r}_gamma_{candidate.gamma:.17g}",
            "context_initialization_seconds": context_seconds,
            "lifting_construction_seconds": lifting_seconds,
            "projected_operator_construction_seconds": projection_seconds,
            "initial_coordinate_fitting_seconds": initial_seconds,
            "reduced_initial_condition_fit_residual": residual,
            "reused_for_lambda_Q_candidates": True,
        },
    )


def _execute_cached_inferred(
    candidate: Figure4Candidate,
    execution: Figure5ExecutionContext,
    cache: InferenceCache,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from .rom import construct_inferred_operators, integrate_selected_rom, regularization_diagnostics

    context = cache.context
    context.config = _config_for_case(execution.config, candidate.rom_case())
    workflow_started = time.perf_counter()
    inference_started = time.perf_counter()
    try:
        construct_inferred_operators(context, regularization_scale=float(TRAINING_COUNT))
    except Exception as error:
        inference_seconds = time.perf_counter() - inference_started
        details = {
            "initialization": "zero_linear_and_quadratic_operators",
            **dict(getattr(context.model, "inference_diagnostics", {}) or {}),
        }
        raise Figure5CaseExecutionError(
            str(error),
            {
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "failed_stage": "operator_inference",
                },
                "solver": {"success": False, "message": "integration not attempted"},
                "inference": details,
                "shared_preparation": cache.preparation,
                "stage_timing": {"inference_seconds": inference_seconds},
                "finite": False,
            },
        ) from error
    inference_seconds = time.perf_counter() - inference_started
    inference = _operator_diagnostics(context)
    inferred_operators_finite = all(
        np.all(np.isfinite(np.asarray(getattr(context.model, attribute))))
        for attribute in (
            "inferredLinear",
            "inferredNonlinear",
            "inferredStreamingLinear",
            "inferredStreamingNonlinear",
        )
    )
    if not inference.get("converged") or not inferred_operators_finite:
        reason = (
            "nonlinear inference did not converge"
            if not inference.get("converged")
            else "nonlinear inference produced non-finite operators"
        )
        raise Figure5CaseExecutionError(
            reason,
            {
                "failure": {
                    "type": "RuntimeError",
                    "message": reason,
                    "failed_stage": "operator_inference",
                },
                "solver": {"success": False, "message": "integration not attempted"},
                "inference": {
                    **inference,
                    "inferred_operators_finite": inferred_operators_finite,
                },
                "shared_preparation": cache.preparation,
                "stage_timing": {"inference_seconds": inference_seconds},
                "finite": False,
            },
        )
    try:
        integration = integrate_selected_rom(context)
    except Exception as error:
        raise Figure5CaseExecutionError(
            str(error),
            {
                "failure": {
                    "type": type(error).__name__,
                    "message": str(error),
                    "failed_stage": "online_integration",
                },
                "solver": dict(getattr(error, "diagnostics", {}) or {}),
                "inference": inference,
                "shared_preparation": cache.preparation,
                "stage_timing": {"inference_seconds": inference_seconds},
                "finite": False,
            },
        ) from error
    online_seconds = float(context.model.last_solve_ivp_elapsed_seconds)
    error_energy, reconstruction_seconds, metric_energy_seconds = _rom_error_energy_chunked(
        context, np.asarray(integration.y)
    )
    aggregate_started = time.perf_counter()
    aggregate_error = relative_space_time_l2_error_from_energies(
        error_energy, execution.reference_energy, context.time
    )
    metric_seconds = metric_energy_seconds + time.perf_counter() - aggregate_started
    regularization = regularization_diagnostics(
        context, regularization_scale=float(TRAINING_COUNT)
    )
    metrics = {
        "metric_id": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION["metric_id"],
        "relative_space_time_l2_error": aggregate_error,
        "online_timing_id": ONLINE_TIMING_DEFINITION["online_timing_id"],
        "fom_integration_elapsed_seconds": execution.fom_integration_elapsed_seconds,
        "rom_online_integration_elapsed_seconds": online_seconds,
        "online_speedup": execution.fom_integration_elapsed_seconds / online_seconds,
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
                "gamma": candidate.gamma,
                "lambda_L": candidate.lambda_L,
                "lambda_Q": candidate.lambda_Q,
            },
            "applied_gram_ridges": {
                "gamma": candidate.applied_gamma_ridge,
                "lambda_L": candidate.applied_lambda_L_ridge,
                "lambda_Q": candidate.applied_lambda_Q_ridge,
            },
            "scaling_count": 1,
            "implementation_diagnostics": regularization,
        },
        "shared_preparation": cache.preparation,
        "stage_timing": {
            "context_initialization_seconds": 0.0,
            "lifting_construction_seconds": 0.0,
            "projected_operator_construction_seconds": 0.0,
            "inference_seconds": inference_seconds,
            "initial_coordinate_fitting_seconds": 0.0,
            "online_integration_seconds": online_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "metric_evaluation_seconds": metric_seconds,
            "total_case_workflow_seconds_before_artifact_write": (
                time.perf_counter() - workflow_started
            ),
        },
        "full_reconstructed_trajectory_constructed": False,
        "reconstruction_chunk_size": 256,
        "finite": bool(
            integration.success
            and float(integration.t[-1]) == 10.0
            and np.isfinite(aggregate_error)
            and np.isfinite(online_seconds)
        ),
    }
    return metrics, diagnostics


def _linear_case(rank: int, operators: str) -> Figure5Case:
    return Figure5Case(
        case_id=f"fig4_linear_{operators}_nr{rank}",
        execution_kind="rom_integration",
        model="linear",
        operators=operators,
        N_r=rank,
        N_q=None,
        reduced_dynamical_dimension=rank,
        projection_dimension=None,
        gamma=None,
        lambda_L=0.0 if operators == "inferred" else None,
        lambda_Q=None,
        applied_lifting_ridge=None,
        applied_linear_ridge=0.0 if operators == "inferred" else None,
        applied_quadratic_ridge=None,
        panel_membership=(operators,),
    )


def _write_final_result(
    *,
    root: Path,
    case: Figure5Case,
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    definition: dict[str, Any],
    execution: Figure5ExecutionContext,
    selected_candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    case_data = case.to_dict()
    if (root / "manifest.json").is_file():
        manifest = _read_json(root / "manifest.json")
        if manifest.get("case_definition_checksum_sha256") != canonical_checksum(case_data):
            raise ValueError(f"final Figure 4 case changed: {case.case_id}")
        if manifest.get("status") == "completed":
            return validate_final_case(root, case)
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "case.json", case_data)
    _write_json(root / "metrics.json", {"schema_version": SCHEMA_VERSION, "metrics": metrics})
    _write_json(
        root / "diagnostics.json",
        {"schema_version": SCHEMA_VERSION, "diagnostics": diagnostics},
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "figure": "Figure 4",
        "case_id": case.case_id,
        "case_definition_checksum_sha256": canonical_checksum(case_data),
        "result_label": RESULT_LABEL,
        "selection_provenance": (
            SELECTION_PROVENANCE if selected_candidate is not None else "not_applicable_linear"
        ),
        "selected_candidate": selected_candidate,
        "search_definition_checksum_sha256": definition["content_checksum_sha256"],
        "benchmark_variant": BENCHMARK_VARIANT,
        "initial_condition_provenance": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
        "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
        "configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
        "catalog_checksum_sha256": execution.catalog.checksum(),
        "metric_definition": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
        "timing_definition": ONLINE_TIMING_DEFINITION,
        "result_files": {
            filename: sha256_file(root / filename)
            for filename in ("case.json", "metrics.json", "diagnostics.json")
        },
        "scientific_execution": {
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "figure5_rerun": False,
            "regularization_search": selected_candidate is not None,
        },
    }
    _write_json(root / "manifest.json", manifest)
    return validate_final_case(root, case)


def validate_final_case(root: Path, case: Figure5Case) -> dict[str, Any]:
    for filename in ("case.json", "manifest.json", "metrics.json", "diagnostics.json"):
        if not (root / filename).is_file():
            raise ValueError(f"final Figure 4 case is missing {filename}")
    if _read_json(root / "case.json") != case.to_dict():
        raise ValueError("final Figure 4 case definition mismatch")
    manifest = _read_json(root / "manifest.json")
    if manifest.get("status") != "completed" or manifest.get(
        "case_definition_checksum_sha256"
    ) != canonical_checksum(case.to_dict()):
        raise ValueError("final Figure 4 manifest is invalid")
    metrics = _read_json(root / "metrics.json")["metrics"]
    for field in (
        "relative_space_time_l2_error",
        "rom_online_integration_elapsed_seconds",
        "online_speedup",
    ):
        value = metrics.get(field)
        if not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"final Figure 4 case has invalid {field}")
    diagnostics = _read_json(root / "diagnostics.json")["diagnostics"]
    if not diagnostics.get("solver", {}).get("success") or diagnostics.get("solver", {}).get(
        "final_time"
    ) != 10.0:
        raise ValueError("final Figure 4 case did not reach t=10")
    return manifest


def _selection_table(records: list[dict[str, Any]], selected_id: str) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda record: (
            record.get("relative_space_time_l2_error")
            if record.get("relative_space_time_l2_error") is not None
            else float("inf"),
            record["candidate"]["candidate_id"],
        ),
    )
    return [
        {
            **record,
            "selected": record["candidate"]["candidate_id"] == selected_id,
            "objective_rank": index + 1 if record.get("admissible") else None,
        }
        for index, record in enumerate(ordered)
    ]


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != _json_safe(value):
            raise ValueError(f"refusing to overwrite changed immutable artifact: {path}")
        return
    _write_json(path, value)


def write_selected_parameters(
    *,
    phase_root: Path,
    run_id: str,
    definition: dict[str, Any],
    selections: dict[str, dict[str, Any]],
) -> tuple[Path, Path]:
    cases: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    for key, selection in sorted(selections.items()):
        candidate = selection["selected_record"]["candidate"]
        case_id = f"fig4_{candidate['model']}_{candidate['operators']}_nr{candidate['N_r']}"
        value = {
            "model": candidate["model"],
            "operator_type": candidate["operators"],
            "N_r": candidate["N_r"],
            "N_q": candidate["N_q"],
            "gamma": candidate["gamma"],
            "lambda_L": candidate["lambda_L"],
            "lambda_Q": candidate["lambda_Q"],
            "applied_ridges": {
                "gamma": candidate["applied_gamma_ridge"],
                "lambda_L": candidate["applied_lambda_L_ridge"],
                "lambda_Q": candidate["applied_lambda_Q_ridge"],
            },
            "selection_metric": selection["selected_record"][
                "relative_space_time_l2_error"
            ],
            "candidate_rank": selection["final_decision"]["candidate_rank"],
            "origin": candidate["origin"],
            "tie_policy_outcome": {
                "tied_candidate_ids": selection["final_decision"]["tied_candidate_ids"],
                "larger_regularization_chosen": selection["final_decision"][
                    "larger_regularization_tie_choice_applied"
                ],
            },
            "search_run_id": run_id,
            "candidate_artifact_path": selection["selected_record"]["artifact_path"],
            "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
            "source_checksum_sha256": definition["source_state"]["sha256"],
            "search_definition_checksum_sha256": definition[
                "content_checksum_sha256"
            ],
            "provenance_status": SELECTION_PROVENANCE,
        }
        cases[case_id] = value
        compact[case_id] = {
            "gamma": candidate["gamma"],
            "lambda_L": candidate["lambda_L"],
            "lambda_Q": candidate["lambda_Q"],
        }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "figure4_selected_parameters",
        "result_label": RESULT_LABEL,
        "provenance_status": SELECTION_PROVENANCE,
        "search_run_id": run_id,
        "catalog_checksum_sha256": definition["catalog_checksum_sha256"],
        "search_definition_checksum_sha256": definition["content_checksum_sha256"],
        "cases": cases,
    }
    artifact = {**payload, "content_checksum_sha256": canonical_checksum(payload)}
    artifact_path = phase_root / "figure4_selected_parameters.json"
    _immutable_json(artifact_path, artifact)
    compact_payload = {
        "schema_version": "1.0.0",
        "catalog_checksum": definition["catalog_checksum_sha256"],
        "selection_objective": "minimum relative_space_time_l2_error_v1 with approved tie and refinement rules",
        "search_result_provenance": {
            "status": SELECTION_PROVENANCE,
            "run_id": run_id,
            "detailed_artifact": str(artifact_path),
            "detailed_artifact_checksum_sha256": artifact["content_checksum_sha256"],
        },
        "author_approval": {
            "status": "author_approved_regenerated_protocol",
            "historical_parameter_recovery": False,
        },
        "cases": compact,
    }
    compact_path = phase_root / "figure4_selected_parameters.proposed.json"
    _immutable_json(compact_path, compact_payload)
    return artifact_path, compact_path


def validate_selected_parameters(
    path: str | Path,
    *,
    expected_run_id: str | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    value = _read_json(Path(path))
    payload = dict(value)
    checksum = payload.pop("content_checksum_sha256", None)
    if checksum != canonical_checksum(payload):
        raise ValueError("selected-parameter content checksum mismatch")
    if value.get("provenance_status") != SELECTION_PROVENANCE:
        raise ValueError("selected parameters have invalid provenance")
    if expected_run_id is not None and value.get("search_run_id") != expected_run_id:
        raise ValueError("selected parameters have the wrong search run")
    if require_complete and len(value.get("cases", {})) != 32:
        raise ValueError("selected parameters require 32 nonlinear final cases")
    if len(value.get("cases", {})) > 32:
        raise ValueError("selected parameters contain too many nonlinear final cases")
    for case in value["cases"].values():
        if case["N_r"] + case["N_q"] != TOTAL_NONLINEAR_DIMENSION:
            raise ValueError("selected nonlinear dimensions do not sum to 564")
        if case["applied_ridges"]["gamma"] != case["gamma"] * TRAINING_COUNT:
            raise ValueError("selected gamma was not scaled exactly once")
        if case["operator_type"] == "inferred":
            if case["applied_ridges"]["lambda_Q"] != case["lambda_Q"] * TRAINING_COUNT:
                raise ValueError("selected lambda_Q was not scaled exactly once")
    return value


def phase8_dry_run_plan(
    *,
    run_id: str,
    source_state: dict[str, Any] | None = None,
    output_root: str | Path = "results/1d/publication",
) -> dict[str, Any]:
    catalog = load_publication_catalog(DEFAULT_CATALOG_PATH)
    definition = search_definition(
        run_id=run_id,
        source_state=source_state or source_state_record(),
        catalog_checksum_sha256=catalog.checksum(),
    )
    root = Path(output_root)
    return {
        "action": "would_execute_or_resume_phase8_figure4",
        "writes_files": False,
        "run_id": run_id,
        "search_definition": definition,
        "search_root": str(root / "figure4_search" / run_id),
        "final_case_root": str(root / "figure4_cases" / run_id),
        "phase_root": str(root / "phase8_runs" / run_id),
        "reuse": {
            "snapshot": True,
            "derivatives": True,
            "pod_basis_coefficients_singular_values": True,
            "phase7_shared_metric_inputs": True,
            "inferred_lifting_per_model_rank_gamma": True,
            "completed_candidates_on_resume": True,
            "selected_candidate_for_final_case": True,
        },
    }


def execute_phase8(
    *,
    run_id: str,
    snapshot_path: str | Path,
    fom_manifest_path: str | Path,
    shared_offline_directory: str | Path,
    shared_metric_inputs_path: str | Path,
    figure5_bundle_path: str | Path,
    output_root: str | Path = "results/1d/publication",
    source_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    phase_root = output_root / "phase8_runs" / run_id
    search_root = output_root / "figure4_search" / run_id / "candidates"
    final_root = output_root / "figure4_cases" / run_id
    phase_root.mkdir(parents=True, exist_ok=True)
    search_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    catalog = load_publication_catalog(DEFAULT_CATALOG_PATH)
    definition = search_definition(
        run_id=run_id,
        source_state=source_state or source_state_record(),
        catalog_checksum_sha256=catalog.checksum(),
    )
    definition_path = phase_root / "search_definition.json"
    if definition_path.is_file():
        existing = _read_json(definition_path)
        validate_search_definition(existing)
        if existing != definition:
            raise ValueError("cannot resume Phase 8 after its search definition changed")
    else:
        _write_json(definition_path, definition)
    validate_search_definition(definition)

    execution, assets = prepare_figure5_execution_context(
        snapshot_path=snapshot_path,
        fom_manifest_path=fom_manifest_path,
        shared_offline_directory=shared_offline_directory,
        phase_root=phase_root,
        shared_metric_inputs_path=shared_metric_inputs_path,
    )
    from .figure5 import validate_figure5_bundle

    figure5_metadata, _ = validate_figure5_bundle(figure5_bundle_path)
    assets.update(
        {
            "phase8_result_label": RESULT_LABEL,
            "search_definition": {
                "path": str(definition_path),
                "sha256": sha256_file(definition_path),
                "content_checksum_sha256": definition["content_checksum_sha256"],
            },
            "reused_figure5_bundle": {
                "path": str(figure5_bundle_path),
                "data_sha256": figure5_metadata["data_file"]["sha256"],
                "rerun": False,
            },
        }
    )
    asset_path = phase_root / "asset_validation.json"
    if not asset_path.is_file():
        _write_json(asset_path, assets)

    outcomes: dict[str, str] = {}
    selections: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []

    for operators in OPERATORS:
        for rank in RANKS:
            case = _linear_case(rank, operators)
            root = final_root / case.case_id
            if (root / "manifest.json").is_file():
                validate_final_case(root, case)
                outcomes[case.case_id] = "reused_completed_result"
                continue
            metrics, diagnostics = _execute_rom_case(case, execution)
            _write_final_result(
                root=root,
                case=case,
                metrics=metrics,
                diagnostics=diagnostics,
                definition=definition,
                execution=execution,
                selected_candidate=None,
            )
            outcomes[case.case_id] = "executed"

    projected_selected: dict[tuple[str, int], dict[str, Any]] = {}
    for model in MODELS:
        for rank in RANKS:
            key = f"projected_{model}_nr{rank}"
            coarse_candidates = [
                _coarse_candidate(
                    model=model,
                    operators="projected",
                    rank=rank,
                    coefficient=value,
                    grid_index=index,
                )
                for index, value in enumerate(GAMMA_GRID)
            ]
            coarse_records = [
                _execute_candidate(
                    candidate=candidate,
                    root=search_root / candidate.candidate_id,
                    definition=definition,
                    execution=execution,
                    runner=lambda candidate=candidate: _execute_rom_case(
                        candidate.rom_case(), execution
                    ),
                )
                for candidate in coarse_candidates
            ]
            try:
                coarse_decision = select_candidate(
                    coarse_records, coefficient_field="gamma"
                )
            except ValueError:
                unresolved.append(key)
                continue
            winner = next(
                candidate
                for candidate in coarse_candidates
                if candidate.candidate_id == coarse_decision["selected_candidate_id"]
            )
            refined_candidates = [
                _refined_candidate(
                    model=model,
                    operators="projected",
                    rank=rank,
                    coefficient=value,
                    neighbor=neighbor,
                )
                for neighbor, value in geometric_refinement_values(
                    GAMMA_GRID, int(winner.grid_index)
                )
            ]
            refined_records = [
                _execute_candidate(
                    candidate=candidate,
                    root=search_root / candidate.candidate_id,
                    definition=definition,
                    execution=execution,
                    runner=lambda candidate=candidate: _execute_rom_case(
                        candidate.rom_case(), execution
                    ),
                )
                for candidate in refined_candidates
            ]
            coarse_winner_record = next(
                record
                for record in coarse_records
                if record["candidate"]["candidate_id"]
                == coarse_decision["selected_candidate_id"]
            )
            final_records = [coarse_winner_record, *refined_records]
            try:
                final_decision = select_candidate(
                    final_records, coefficient_field="gamma"
                )
            except ValueError:
                unresolved.append(key)
                continue
            selected_record = next(
                record
                for record in final_records
                if record["candidate"]["candidate_id"]
                == final_decision["selected_candidate_id"]
            )
            selection = {
                "coarse_decision": coarse_decision,
                "final_decision": final_decision,
                "candidate_table": _selection_table(
                    [*coarse_records, *refined_records],
                    final_decision["selected_candidate_id"],
                ),
                "selected_record": selected_record,
            }
            selections[key] = selection
            projected_selected[(model, rank)] = selection
            _write_json(
                phase_root / "search_index.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running",
                    "selections": selections,
                    "unresolved": unresolved,
                },
            )

    for model in MODELS:
        for rank in RANKS:
            key = f"inferred_{model}_nr{rank}"
            projected = projected_selected.get((model, rank))
            if projected is None:
                unresolved.append(key)
                continue
            gamma = float(projected["selected_record"]["candidate"]["gamma"])
            coarse_candidates = [
                _coarse_candidate(
                    model=model,
                    operators="inferred",
                    rank=rank,
                    coefficient=value,
                    grid_index=index,
                    gamma=gamma,
                )
                for index, value in enumerate(LAMBDA_Q_GRID)
            ]
            cache: list[InferenceCache] = []

            def cached_runner(candidate: Figure4Candidate):
                if not cache:
                    cache.append(_prepare_inference_cache(candidate, execution))
                return _execute_cached_inferred(candidate, execution, cache[0])

            coarse_records = [
                _execute_candidate(
                    candidate=candidate,
                    root=search_root / candidate.candidate_id,
                    definition=definition,
                    execution=execution,
                    runner=lambda candidate=candidate: cached_runner(candidate),
                )
                for candidate in coarse_candidates
            ]
            try:
                coarse_decision = select_candidate(
                    coarse_records, coefficient_field="lambda_Q"
                )
            except ValueError:
                unresolved.append(key)
                continue
            winner = next(
                candidate
                for candidate in coarse_candidates
                if candidate.candidate_id == coarse_decision["selected_candidate_id"]
            )
            refined_candidates = [
                _refined_candidate(
                    model=model,
                    operators="inferred",
                    rank=rank,
                    coefficient=value,
                    neighbor=neighbor,
                    gamma=gamma,
                )
                for neighbor, value in geometric_refinement_values(
                    LAMBDA_Q_GRID, int(winner.grid_index)
                )
            ]
            refined_records = [
                _execute_candidate(
                    candidate=candidate,
                    root=search_root / candidate.candidate_id,
                    definition=definition,
                    execution=execution,
                    runner=lambda candidate=candidate: cached_runner(candidate),
                )
                for candidate in refined_candidates
            ]
            coarse_winner_record = next(
                record
                for record in coarse_records
                if record["candidate"]["candidate_id"]
                == coarse_decision["selected_candidate_id"]
            )
            final_records = [coarse_winner_record, *refined_records]
            try:
                final_decision = select_candidate(
                    final_records, coefficient_field="lambda_Q"
                )
            except ValueError:
                unresolved.append(key)
                continue
            selected_record = next(
                record
                for record in final_records
                if record["candidate"]["candidate_id"]
                == final_decision["selected_candidate_id"]
            )
            selections[key] = {
                "coarse_decision": coarse_decision,
                "final_decision": final_decision,
                "gamma_source_selection": projected["final_decision"],
                "gamma_reuse_rationale": (
                    "gamma regularizes the shared learned lifting independently of "
                    "subsequent projected or inferred streaming construction"
                ),
                "candidate_table": _selection_table(
                    [*coarse_records, *refined_records],
                    final_decision["selected_candidate_id"],
                ),
                "selected_record": selected_record,
            }
            _write_json(
                phase_root / "search_index.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "running",
                    "selections": selections,
                    "unresolved": unresolved,
                },
            )

    selected_path, proposed_path = write_selected_parameters(
        phase_root=phase_root,
        run_id=run_id,
        definition=definition,
        selections=selections,
    )
    selected_parameters = validate_selected_parameters(
        selected_path, expected_run_id=run_id, require_complete=False
    )
    for case_id, selected in selected_parameters["cases"].items():
        candidate = next(
            selection["selected_record"]
            for selection in selections.values()
            if selection["selected_record"]["candidate"]["candidate_id"]
            == Path(selected["candidate_artifact_path"]).name
        )
        candidate_root = Path(candidate["artifact_path"])
        candidate_data = candidate["candidate"]
        case = Figure5Case(
            case_id=case_id,
            execution_kind="rom_integration",
            model=candidate_data["model"],
            operators=candidate_data["operators"],
            N_r=candidate_data["N_r"],
            N_q=candidate_data["N_q"],
            reduced_dynamical_dimension=candidate_data["N_r"],
            projection_dimension=None,
            gamma=candidate_data["gamma"],
            lambda_L=candidate_data["lambda_L"],
            lambda_Q=candidate_data["lambda_Q"],
            applied_lifting_ridge=candidate_data["applied_gamma_ridge"],
            applied_linear_ridge=candidate_data["applied_lambda_L_ridge"],
            applied_quadratic_ridge=candidate_data["applied_lambda_Q_ridge"],
            panel_membership=(candidate_data["operators"],),
        )
        _write_final_result(
            root=final_root / case_id,
            case=case,
            metrics=_read_json(candidate_root / "metrics.json")["metrics"],
            diagnostics=_read_json(candidate_root / "diagnostics.json")["diagnostics"],
            definition=definition,
            execution=execution,
            selected_candidate={
                "candidate_id": candidate_data["candidate_id"],
                "artifact_path": str(candidate_root),
                "manifest_sha256": sha256_file(candidate_root / "manifest.json"),
                "metrics_sha256": sha256_file(candidate_root / "metrics.json"),
                "diagnostics_sha256": sha256_file(candidate_root / "diagnostics.json"),
            },
        )
        outcomes[case_id] = "promoted_selected_candidate"

    expected_final = 48
    completed_final = len(list(final_root.glob("*/manifest.json")))
    status = "complete" if not unresolved and completed_final == expected_final else "partial"
    index = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "run_id": run_id,
        "result_label": RESULT_LABEL,
        "search_definition_checksum_sha256": definition["content_checksum_sha256"],
        "selections": selections,
        "selected_parameter_artifact": {
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
            "content_checksum_sha256": selected_parameters["content_checksum_sha256"],
        },
        "proposed_configuration": {
            "path": str(proposed_path),
            "sha256": sha256_file(proposed_path),
        },
        "final_case_root": str(final_root),
        "completed_final_cases": completed_final,
        "expected_final_cases": expected_final,
        "unresolved": unresolved,
        "outcomes": outcomes,
    }
    _write_json(phase_root / "search_index.json", index)
    return index


def build_figure4_bundle(
    *,
    run_id: str,
    output_directory: str | Path,
    output_root: str | Path = "results/1d/publication",
    allow_partial: bool = False,
) -> Path:
    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"Figure 4 bundle directory already exists: {output}")
    output_root = Path(output_root)
    final_root = output_root / "figure4_cases" / run_id
    phase_root = output_root / "phase8_runs" / run_id
    index = _read_json(phase_root / "search_index.json")
    selected = validate_selected_parameters(
        phase_root / "figure4_selected_parameters.json",
        expected_run_id=run_id,
        require_complete=not allow_partial,
    )
    arrays: dict[str, np.ndarray] = {
        "n_r": np.asarray(RANKS, dtype=int),
        "n_q": np.asarray([TOTAL_NONLINEAR_DIMENSION - rank for rank in RANKS], dtype=int),
    }
    records: dict[str, Any] = {}
    missing: list[str] = []
    for operators in OPERATORS:
        for model in ("linear", *MODELS):
            errors: list[float] = []
            speedups: list[float] = []
            online: list[float] = []
            gamma: list[float] = []
            lambda_q: list[float] = []
            for rank in RANKS:
                case_id = f"fig4_{model}_{operators}_nr{rank}"
                root = final_root / case_id
                try:
                    case_data = _read_json(root / "case.json")
                    case = Figure5Case(
                        case_id=case_data["case_id"],
                        execution_kind=case_data["execution_kind"],
                        model=case_data["model"],
                        operators=case_data["operators"],
                        N_r=case_data["N_r"],
                        N_q=case_data["N_q"],
                        reduced_dynamical_dimension=case_data["reduced_dynamical_dimension"],
                        projection_dimension=case_data["projection_dimension"],
                        gamma=case_data["gamma"],
                        lambda_L=case_data["lambda_L"],
                        lambda_Q=case_data["lambda_Q"],
                        applied_lifting_ridge=case_data["applied_lifting_ridge"],
                        applied_linear_ridge=case_data["applied_linear_ridge"],
                        applied_quadratic_ridge=case_data["applied_quadratic_ridge"],
                        panel_membership=tuple(case_data["panel_membership"]),
                    )
                    validate_final_case(root, case)
                    metrics = _read_json(root / "metrics.json")["metrics"]
                    diagnostics = _read_json(root / "diagnostics.json")["diagnostics"]
                except (FileNotFoundError, KeyError, TypeError, ValueError):
                    missing.append(case_id)
                    errors.append(np.nan)
                    speedups.append(np.nan)
                    online.append(np.nan)
                    if model != "linear":
                        gamma.append(np.nan)
                        if operators == "inferred":
                            lambda_q.append(np.nan)
                    continue
                errors.append(float(metrics["relative_space_time_l2_error"]))
                speedups.append(float(metrics["online_speedup"]))
                online.append(float(metrics["rom_online_integration_elapsed_seconds"]))
                if model != "linear":
                    gamma.append(float(case.gamma))
                    if operators == "inferred":
                        lambda_q.append(float(case.lambda_Q))
                records[case_id] = {
                    "case": case.to_dict(),
                    "result_path": str(root),
                    "manifest_sha256": sha256_file(root / "manifest.json"),
                    "metrics_sha256": sha256_file(root / "metrics.json"),
                    "diagnostics_sha256": sha256_file(root / "diagnostics.json"),
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                }
            prefix = f"{operators}__{model}"
            arrays[prefix + "__error"] = np.asarray(errors)
            arrays[prefix + "__online_speedup"] = np.asarray(speedups)
            arrays[prefix + "__online_seconds"] = np.asarray(online)
            if model != "linear":
                arrays[prefix + "__gamma"] = np.asarray(gamma)
                if operators == "inferred":
                    arrays[prefix + "__lambda_Q"] = np.asarray(lambda_q)
    complete = not missing and len(records) == 48 and index.get("status") == "complete"
    if not complete and not allow_partial:
        raise ValueError("Figure 4 bundle requires all 48 final cases")
    output.mkdir(parents=True, exist_ok=False)
    data_path = output / "figure4_data.npz"
    np.savez_compressed(data_path, **arrays)
    definition_path = phase_root / "search_definition.json"
    selected_path = phase_root / "figure4_selected_parameters.json"
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "figure": "Figure 4",
        "run_id": run_id,
        "result_label": RESULT_LABEL,
        "selection_provenance": SELECTION_PROVENANCE,
        "benchmark_variant": BENCHMARK_VARIANT,
        "initial_condition_provenance": AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
        "case_set_status": "complete" if complete else "partial",
        "complete_publication_reproduction": False,
        "missing_case_ids": missing,
        "metric_definition": RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION,
        "timing_definition": ONLINE_TIMING_DEFINITION,
        "N_r_values": list(RANKS),
        "N_q_values": [TOTAL_NONLINEAR_DIMENSION - rank for rank in RANKS],
        "case_records": records,
        "dataset_checksum_sha256": EXPECTED_DATASET_CHECKSUM,
        "configuration_checksum_sha256": LEGACY_CONFIG_CHECKSUM,
        "catalog_checksum_sha256": load_publication_catalog().checksum(),
        "search_definition": {
            "path": str(definition_path),
            "sha256": sha256_file(definition_path),
            "content_checksum_sha256": _read_json(definition_path)[
                "content_checksum_sha256"
            ],
        },
        "selected_parameters": {
            "path": str(selected_path),
            "sha256": sha256_file(selected_path),
            "content_checksum_sha256": selected["content_checksum_sha256"],
        },
        "array_metadata": [
            {"name": name, "shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in sorted(arrays.items())
        ],
        "scientific_execution": {
            "fom_run": False,
            "derivatives_recomputed": False,
            "pod_svd_recomputed": False,
            "figure5_rerun": False,
        },
        "data_file": {"path": str(data_path), "sha256": sha256_file(data_path)},
    }
    _write_json(output / "figure4_data.json", metadata)
    from .figure4_plotting import validate_figure4_bundle

    validate_figure4_bundle(output, require_complete=not allow_partial)
    return output
