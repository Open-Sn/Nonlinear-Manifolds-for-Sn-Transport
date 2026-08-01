#!/usr/bin/env python3
"""Diagnose the four Phase-5 nonlinear failures without recomputing FOM/POD data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import scipy.integrate


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from Nonlinear_Manifold_ROM import (  # noqa: E402
    NonlinearManifoldReducedModel,
    ReducedIntegrationError,
    reduced_integration_diagnostics,
)
from one_d.config import load_config  # noqa: E402
from one_d.nonlinear_diagnostics import (  # noqa: E402
    difference_metrics,
    historical_alternating_inference,
    historical_initial_coordinate,
    historical_lifting,
    historical_nonlinear_rhs,
    historical_projected_operators,
    historical_quadratic_features,
    latent_derivatives,
)
from one_d.problem import assemble_operators, build_problem  # noqa: E402
from one_d.publication_artifacts import sha256_file  # noqa: E402
from one_d.publication_experiments import (  # noqa: E402
    load_publication_catalog,
    resolve_case_configuration,
)
from one_d.rom import initialize_rom_context_from_precomputed  # noqa: E402
from one_d.shared_offline import load_shared_offline_artifacts  # noqa: E402


SNAPSHOT_SHA256 = "a3885dc5a071f67afb514e3d130d15cd993737a174313084f7e1ed0911cef6b3"
FAILED_CASES = (
    "fig2_elementwise_projected",
    "fig2_tensorial_projected",
    "fig3_elementwise_inferred",
    "fig3_tensorial_inferred",
)
MODELS = ("elementwise", "tensorial")
CHECKPOINTS = (1, 10, 100, 868, 1000, 10000, 23367, 50000, 100000)
HORIZONS = (0.001, 0.01, 0.1, 1.0, 2.5, 7.5, 10.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--shared-offline", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


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


def _matrix_summary(matrix: np.ndarray, *, square_spectrum: bool = False) -> dict[str, Any]:
    matrix = np.asarray(matrix)
    singular = np.linalg.svd(matrix, compute_uv=False)
    summary: dict[str, Any] = {
        "shape": list(matrix.shape),
        "frobenius_norm": float(np.linalg.norm(matrix)),
        "spectral_norm": float(singular[0]) if singular.size else 0.0,
        "smallest_singular_value": float(singular[-1]) if singular.size else 0.0,
        "condition_estimate": (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0.0
            else None
        ),
        "finite": bool(np.all(np.isfinite(matrix))),
    }
    if square_spectrum and matrix.shape[0] == matrix.shape[1]:
        eigenvalues = np.linalg.eigvals(matrix)
        summary["eigenvalue_real_part_minimum"] = float(np.min(eigenvalues.real))
        summary["eigenvalue_real_part_maximum"] = float(np.max(eigenvalues.real))
        summary["eigenvalue_imaginary_magnitude_maximum"] = float(
            np.max(np.abs(eigenvalues.imag))
        )
    return summary


def _mass_norm(vector: np.ndarray, mass: Any) -> float:
    vector = np.asarray(vector)
    return float(np.sqrt(max(float(vector @ mass.dot(vector)), 0.0)))


def _finite_difference_jacobian(rhs, coordinate: np.ndarray) -> dict[str, Any]:
    coordinate = np.asarray(coordinate, dtype=float)
    jacobian = np.empty((coordinate.size, coordinate.size))
    for index in range(coordinate.size):
        step = np.sqrt(np.finfo(float).eps) * max(1.0, abs(coordinate[index]))
        forward = coordinate.copy()
        backward = coordinate.copy()
        forward[index] += step
        backward[index] -= step
        jacobian[:, index] = (rhs(forward) - rhs(backward)) / (2.0 * step)
    eigenvalues = np.linalg.eigvals(jacobian)
    return {
        "frobenius_norm": float(np.linalg.norm(jacobian)),
        "spectral_norm": float(np.linalg.norm(jacobian, 2)),
        "eigenvalue_real_part_minimum": float(np.min(eigenvalues.real)),
        "eigenvalue_real_part_maximum": float(np.max(eigenvalues.real)),
        "eigenvalue_imaginary_magnitude_maximum": float(
            np.max(np.abs(eigenvalues.imag))
        ),
        "finite": bool(np.all(np.isfinite(jacobian))),
    }


def _state_rhs_diagnostics(
    coordinate: np.ndarray,
    *,
    linear_operator: np.ndarray,
    nonlinear_operator: np.ndarray,
    latent_basis: np.ndarray,
    nonlinear_basis: np.ndarray,
    steady_state: np.ndarray,
    model: str,
) -> dict[str, Any]:
    coordinate = np.asarray(coordinate)
    features = historical_quadratic_features(coordinate, model)
    linear_rhs = -linear_operator @ coordinate
    lifted_rhs = -nonlinear_operator @ features
    total_rhs = linear_rhs + lifted_rhs
    reconstruction = (
        steady_state
        + latent_basis @ coordinate
        + nonlinear_basis @ features
    )
    rhs = lambda value: historical_nonlinear_rhs(
        value, linear_operator, nonlinear_operator, model
    )
    return {
        "latent_state_norm": float(np.linalg.norm(coordinate)),
        "linear_rhs_norm": float(np.linalg.norm(linear_rhs)),
        "lifted_rhs_norm": float(np.linalg.norm(lifted_rhs)),
        "total_rhs_norm": float(np.linalg.norm(total_rhs)),
        "reconstructed_state_norm": float(np.linalg.norm(reconstruction)),
        "finite": bool(
            np.all(np.isfinite(coordinate))
            and np.all(np.isfinite(total_rhs))
            and np.all(np.isfinite(reconstruction))
        ),
        "jacobian": _finite_difference_jacobian(rhs, coordinate),
    }


def _solve_direct(
    rhs,
    initial: np.ndarray,
    time_values: np.ndarray,
    horizon: float,
    config,
) -> Any:
    selected = time_values[time_values <= horizon + 1.0e-12]
    return scipy.integrate.solve_ivp(
        fun=lambda _time, state: rhs(state),
        t_span=(config.time.initial_time, float(horizon)),
        y0=np.asarray(initial),
        method=config.time.rom_method,
        atol=config.time.rom_absolute_tolerance,
        rtol=config.time.rom_relative_tolerance,
        t_eval=selected,
    )


def _public_inference(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key
        not in {
            "linear_operator",
            "nonlinear_operator",
            "rhs_linear",
            "rhs_nonlinear",
        }
    }


def _inference_variant(
    *,
    name: str,
    count: int,
    defective: bool,
    semantics: str,
    model: str,
    gamma: float,
    lambda_linear: float,
    lambda_quadratic: float,
    latent_coefficients: np.ndarray,
    lifting_coefficients: np.ndarray,
    shared_projected_derivative: np.ndarray,
    latent_absorption_minus_scattering: np.ndarray,
    lifting_basis: np.ndarray,
    latent_basis: np.ndarray,
    operators: Any,
    dt: float,
    tolerance: float,
    maximum_iterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    latent = np.asarray(latent_coefficients[:, :count])
    lifting_target = np.asarray(lifting_coefficients[:, :count])
    scale = count if semantics == "paper_scaled" else 1
    lift = historical_lifting(latent, lifting_target, gamma * scale, model)
    nonlinear_basis = lifting_basis @ lift["lifting_matrix"]
    absorption_cross = latent_basis.T @ operators.total_interaction @ nonlinear_basis
    scattering_cross = latent_basis.T @ operators.scattering @ nonlinear_basis
    if name == "A_current_corrected":
        derivative = np.asarray(shared_projected_derivative)
    else:
        derivative = latent_derivatives(
            latent,
            dt,
            historical_defect=defective,
        )
    residual = (
        derivative
        + latent_absorption_minus_scattering @ latent
        + (absorption_cross - scattering_cross) @ lift["features"]
    )
    inferred = historical_alternating_inference(
        latent,
        lift["features"],
        residual,
        ridge_linear=lambda_linear * scale,
        ridge_nonlinear=lambda_quadratic * scale,
        tolerance=tolerance,
        maximum_iterations=maximum_iterations,
        checkpoints=CHECKPOINTS,
    )
    public = _public_inference(inferred)
    public.update(
        {
            "variant": name,
            "non_authoritative_provenance_diagnostic": name
            != "A_current_corrected",
            "training_snapshot_count": count,
            "historical_derivative_defect": defective,
            "regularization_semantics": semantics,
            "lifting_ridge_actual": gamma * scale,
            "linear_inference_ridge_actual": lambda_linear * scale,
            "quadratic_inference_ridge_actual": lambda_quadratic * scale,
            "derivative_matrix": _matrix_summary(derivative),
            "feature_matrix_shape": list(lift["features"].shape),
            "inference_residual_matrix": _matrix_summary(residual),
        }
    )
    internal = {
        "linear_operator": inferred["linear_operator"],
        "nonlinear_operator": inferred["nonlinear_operator"],
        "lifting_matrix": lift["lifting_matrix"],
        "derivative": derivative,
    }
    return public, internal


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "action": "would_diagnose",
                    "failed_cases": list(FAILED_CASES),
                    "reruns_fom": False,
                    "recomputes_svd": False,
                    "tunes_parameters": False,
                    "writes_files": False,
                },
                indent=2,
            )
        )
        return 0

    output = Path(args.output_directory)
    if output.exists():
        raise FileExistsError(f"diagnostic output already exists: {output}")
    output.mkdir(parents=True)
    report_path = output / "diagnostic_report.json"
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "running",
        "creation_time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "failed_cases": list(FAILED_CASES),
            "FOM_rerun": False,
            "full_SVD_recomputed": False,
            "regularization_search": False,
            "figure_4_or_5_executed": False,
        },
    }
    _write_json(report_path, report)
    try:
        base_config = load_config("configs/1d/legacy_production.json")
        catalog = load_publication_catalog()
        snapshot_path = Path(args.snapshot)
        if sha256_file(snapshot_path) != SNAPSHOT_SHA256:
            raise ValueError("production snapshot checksum mismatch")
        shared_root = Path(args.shared_offline)
        shared = load_shared_offline_artifacts(
            shared_root,
            base_config,
            dataset_sha256=SNAPSHOT_SHA256,
        )
        snapshot = np.load(snapshot_path, mmap_mode="r", allow_pickle=False)
        problem = build_problem(base_config)
        operators = assemble_operators(problem)
        shared_manifest = json.loads(
            (shared_root / "manifest.json").read_text(encoding="utf-8")
        )
        report["provenance"] = {
            "snapshot_path": str(snapshot_path),
            "snapshot_sha256": SNAPSHOT_SHA256,
            "shared_offline_path": str(shared_root),
            "shared_offline_manifest_sha256": sha256_file(
                shared_root / "manifest.json"
            ),
            "shared_arrays": shared_manifest["arrays"],
            "legacy_configuration_checksum": base_config.checksum(),
            "publication_catalog_checksum": catalog.checksum(),
            "historical_reference_commit": "bff6910",
            "historical_reference_reason": (
                "corrected inclusive training partition and backward derivative stencils; "
                "immediately predates Phase-3 workflow extraction"
            ),
        }
        count = int(shared.training_indices.size)
        report["regularization_scaling"] = {
            "training_snapshot_count": count,
            "paper_formulas": {
                "lifting": "W W.T + gamma N_s I",
                "linear_inference": "S S.T + lambda_L N_s I",
                "quadratic_inference": "H H.T + lambda_Q N_s I",
            },
            "failed_phase5_publication_execution": (
                "catalog values are passed directly and added once; no N_s multiplication"
            ),
            "historical_driver": {
                "parameter_name_ambiguity": (
                    "lambda_E/lambda_H are actual ridge terms even though the underlying "
                    "base coefficients were multiplied by an integer training count by main()"
                ),
                "caller_training_count_used_for_scaling": 7500,
                "lifting_ridge": 0.0001875,
                "tensorial_quadratic_inference_ridge": 0.012,
                "elementwise_quadratic_inference_ridge": 12.0,
            },
            "double_scaling_detected": False,
            "missing_scaling_detected_for_publication_formulas": True,
            "cases": {},
        }
        for case_id in FAILED_CASES:
            case = catalog.get(case_id)
            report["regularization_scaling"]["cases"][case_id] = {
                "catalog_gamma": case.lifting_regularization_gamma,
                "catalog_lambda_L": case.lambda_L,
                "catalog_lambda_Q": case.lambda_Q,
                "current_lifting_ridge_actual": case.lifting_regularization_gamma,
                "paper_lifting_ridge_actual": (
                    None
                    if case.lifting_regularization_gamma is None
                    else case.lifting_regularization_gamma * count
                ),
                "current_linear_inference_ridge_actual": case.lambda_L,
                "paper_linear_inference_ridge_actual": (
                    None if case.lambda_L is None else case.lambda_L * count
                ),
                "current_quadratic_inference_ridge_actual": case.lambda_Q,
                "paper_quadratic_inference_ridge_actual": (
                    None if case.lambda_Q is None else case.lambda_Q * count
                ),
            }

        latent_basis = np.asarray(shared.basis[:, :16])
        lifting_basis = np.asarray(shared.basis[:, 16:564])
        latent_coefficients = np.asarray(shared.coefficients[:16, :])
        lifting_coefficients = np.asarray(shared.coefficients[16:564, :])
        report["basis_selection"] = {
            "latent_basis_shape": list(latent_basis.shape),
            "lifting_basis_shape": list(lifting_basis.shape),
            "latent_columns": [0, 15],
            "lifting_columns": [16, 563],
            "N_r": 16,
            "N_q": 548,
            "basis_block_overlap": False,
        }
        model_reports: dict[str, Any] = {}
        internal_models: dict[str, dict[str, Any]] = {}
        for model_name in MODELS:
            print(f"diagnosing {model_name} lifting/projection/initial state", flush=True)
            projected_case = catalog.get(f"fig2_{model_name}_projected")
            inferred_case = catalog.get(f"fig3_{model_name}_inferred")
            config = resolve_case_configuration(projected_case)
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
                model_name=model_name,
                operator_choice="projected",
                problem=problem,
                operators=operators,
            )
            current = context.model
            gamma = float(projected_case.lifting_regularization_gamma)
            raw_lift = historical_lifting(
                latent_coefficients, lifting_coefficients, gamma, model_name
            )
            scaled_lift = historical_lifting(
                latent_coefficients,
                lifting_coefficients,
                gamma * count,
                model_name,
            )
            current.compute_nonlinear_embedding(lambda_E=gamma)
            current.compute_projected_operators()
            current_raw_lifting = np.asarray(current.nonlinear_lift_matrix)
            raw_nonlinear_basis = np.asarray(current.pod_nonlinear_basis)
            scaled_nonlinear_basis = lifting_basis @ scaled_lift["lifting_matrix"]
            raw_reference_operators = historical_projected_operators(
                latent_basis,
                raw_nonlinear_basis,
                streaming=operators.streaming,
                absorption=operators.total_interaction,
                scattering=operators.scattering,
            )
            scaled_reference_operators = historical_projected_operators(
                latent_basis,
                scaled_nonlinear_basis,
                streaming=operators.streaming,
                absorption=operators.total_interaction,
                scattering=operators.scattering,
            )
            current_operators = {
                "G_r": np.asarray(current.projectedStreamingLinear),
                "A_r": np.asarray(current.projectedAbsorptionLinear),
                "B_r": np.asarray(current.projectedScatteringLinear),
                "G_lifted": np.asarray(current.projectedStreamingNonlinear),
                "A_lifted": np.asarray(current.projectedAbsorptionNonlinear),
                "B_lifted": np.asarray(current.projectedScatteringNonlinear),
                "combined_linear": np.asarray(current.projectedLinear),
                "combined_lifted": np.asarray(current.projectedNonlinear),
            }
            feature_gram_unregularized = (
                raw_lift["features"] @ raw_lift["features"].T
            )
            raw_residual = lifting_coefficients - raw_lift["lifted_coefficients"]
            scaled_residual = (
                lifting_coefficients - scaled_lift["lifted_coefficients"]
            )
            lifting_report = {
                "latent_snapshot_shape": list(latent_coefficients.shape),
                "lifting_snapshot_shape": list(lifting_coefficients.shape),
                "feature_shape": list(raw_lift["features"].shape),
                "feature_ordering": (
                    "c_i squared only"
                    if model_name == "elementwise"
                    else "row-major numpy.triu_indices pairs c_i*c_j for i<=j"
                ),
                "feature_pair_count": int(raw_lift["features"].shape[0]),
                "current_actual_ridge": gamma,
                "paper_formula_actual_ridge": gamma * count,
                "unregularized_gram": _matrix_summary(feature_gram_unregularized),
                "current_regularized_gram": _matrix_summary(raw_lift["gram"]),
                "paper_scaled_regularized_gram": _matrix_summary(
                    scaled_lift["gram"]
                ),
                "regression_rhs": _matrix_summary(raw_lift["regression_rhs"]),
                "current_lifting_matrix": _matrix_summary(
                    current_raw_lifting
                ),
                "paper_scaled_lifting_matrix": _matrix_summary(
                    scaled_lift["lifting_matrix"]
                ),
                "current_vs_historical_same_ridge": difference_metrics(
                    current_raw_lifting, raw_lift["lifting_matrix"]
                ),
                "current_vs_paper_scaled": difference_metrics(
                    current_raw_lifting, scaled_lift["lifting_matrix"]
                ),
                "current_training_residual_frobenius": float(
                    np.linalg.norm(raw_residual)
                ),
                "current_training_residual_relative": float(
                    np.linalg.norm(raw_residual)
                    / np.linalg.norm(lifting_coefficients)
                ),
                "paper_scaled_training_residual_frobenius": float(
                    np.linalg.norm(scaled_residual)
                ),
                "paper_scaled_training_residual_relative": float(
                    np.linalg.norm(scaled_residual)
                    / np.linalg.norm(lifting_coefficients)
                ),
            }
            operator_report: dict[str, Any] = {
                "sign_convention": "G + A - B for latent and lifted terms",
                "full_mass_convention": (
                    "same spatial DG mass repeated by angle without angular weights"
                ),
                "current": {},
                "current_vs_historical_same_ridge": {},
                "current_vs_paper_scaled": {},
            }
            for name, value in current_operators.items():
                operator_report["current"][name] = _matrix_summary(
                    value, square_spectrum=name == "combined_linear"
                )
                operator_report["current_vs_historical_same_ridge"][name] = (
                    difference_metrics(value, raw_reference_operators[name])
                )
                operator_report["current_vs_paper_scaled"][name] = (
                    difference_metrics(value, scaled_reference_operators[name])
                )

            simple_initial = latent_coefficients[:, 0].copy()
            raw_initial_result, raw_initial_diagnostics = historical_initial_coordinate(
                simple_initial,
                np.asarray(shared.coefficients[:564, 0]),
                raw_lift["lifting_matrix"],
                model_name,
            )
            scaled_initial_result, scaled_initial_diagnostics = (
                historical_initial_coordinate(
                    simple_initial,
                    np.asarray(shared.coefficients[:564, 0]),
                    scaled_lift["lifting_matrix"],
                    model_name,
                )
            )
            current.compute_initial_conditions()
            current_initial = np.asarray(current.initial_condition).copy()
            target_centered = np.asarray(snapshot[:, 0]) - np.asarray(shared.steady_state)

            def reconstruction_residual(
                coordinate: np.ndarray,
                nonlinear_basis: np.ndarray,
            ) -> dict[str, float]:
                reconstruction = (
                    latent_basis @ coordinate
                    + nonlinear_basis
                    @ historical_quadratic_features(coordinate, model_name)
                )
                difference = target_centered - reconstruction
                return {
                    "euclidean": float(np.linalg.norm(difference)),
                    "mass": _mass_norm(difference, operators.mass),
                }

            raw_initial_diagnostics.update(
                {
                    "optimizer_method": "Nelder-Mead",
                    "optimizer_tolerance": 1.0e-7,
                    "optimizer_maximum_iterations": 10000,
                    "optimizer_success": bool(raw_initial_result.success),
                    "optimizer_message": str(raw_initial_result.message),
                    "optimizer_iterations": int(raw_initial_result.nit),
                    "optimizer_function_evaluations": int(raw_initial_result.nfev),
                    "full_reconstruction_residual": reconstruction_residual(
                        raw_initial_result.x, raw_nonlinear_basis
                    ),
                    "current_vs_historical_coordinate": difference_metrics(
                        current_initial, raw_initial_result.x
                    ),
                    "simple_latent_projection_norm": float(
                        np.linalg.norm(simple_initial)
                    ),
                    "initial_full_centered_state_euclidean_norm": float(
                        np.linalg.norm(target_centered)
                    ),
                    "initial_full_centered_state_mass_norm": _mass_norm(
                        target_centered, operators.mass
                    ),
                }
            )
            scaled_initial_diagnostics.update(
                {
                    "optimizer_success": bool(scaled_initial_result.success),
                    "optimizer_message": str(scaled_initial_result.message),
                    "optimizer_iterations": int(scaled_initial_result.nit),
                    "optimizer_function_evaluations": int(
                        scaled_initial_result.nfev
                    ),
                    "full_reconstruction_residual": reconstruction_residual(
                        scaled_initial_result.x, scaled_nonlinear_basis
                    ),
                }
            )
            raw_linear_operator = current_operators["combined_linear"]
            raw_nonlinear_operator = current_operators["combined_lifted"]
            scaled_linear_operator = scaled_reference_operators["combined_linear"]
            scaled_nonlinear_operator = scaled_reference_operators["combined_lifted"]
            raw_rhs = lambda value: historical_nonlinear_rhs(
                value,
                raw_linear_operator,
                raw_nonlinear_operator,
                model_name,
            )
            scaled_rhs = lambda value: historical_nonlinear_rhs(
                value,
                scaled_linear_operator,
                scaled_nonlinear_operator,
                model_name,
            )
            rhs_points = {
                "optimized_initial": current_initial,
                "simple_linear_projection": simple_initial,
                "zero": np.zeros(16),
                "deterministic_ramp": np.linspace(-0.1, 0.1, 16),
                "deterministic_alternating": 0.05
                * np.where(np.arange(16) % 2, -1.0, 1.0),
            }
            rhs_report: dict[str, Any] = {}
            for name, coordinate in rhs_points.items():
                current_value = (
                    -current.projectedLinear @ coordinate
                    - current.projectedNonlinear
                    @ current.nonlinear_function(coordinate)
                )
                historical_value = raw_rhs(coordinate)
                rhs_report[name] = {
                    "current": _state_rhs_diagnostics(
                        coordinate,
                        linear_operator=raw_linear_operator,
                        nonlinear_operator=raw_nonlinear_operator,
                        latent_basis=latent_basis,
                        nonlinear_basis=raw_nonlinear_basis,
                        steady_state=np.asarray(shared.steady_state),
                        model=model_name,
                    ),
                    "current_vs_historical": difference_metrics(
                        current_value, historical_value
                    ),
                }

            print(f"rerunning unchanged failed {model_name} Radau case", flush=True)
            full_failure = None
            returned_last_state = None
            try:
                result = current.integrate_reduced(
                    intrusive=True,
                    method=config.time.rom_method,
                    atol=config.time.rom_absolute_tolerance,
                    rtol=config.time.rom_relative_tolerance,
                    initial_time=config.time.initial_time,
                )
                full_failure = reduced_integration_diagnostics(
                    result, config.time.final_time
                )
                if result.y.shape[1]:
                    returned_last_state = result.y[:, -1]
            except ReducedIntegrationError as error:
                full_failure = error.diagnostics
                if error.result.y.shape[1]:
                    returned_last_state = error.result.y[:, -1]
            if returned_last_state is not None:
                rhs_report["last_returned_before_failure"] = {
                    "current": _state_rhs_diagnostics(
                        returned_last_state,
                        linear_operator=raw_linear_operator,
                        nonlinear_operator=raw_nonlinear_operator,
                        latent_basis=latent_basis,
                        nonlinear_basis=raw_nonlinear_basis,
                        steady_state=np.asarray(shared.steady_state),
                        model=model_name,
                    ),
                    "current_vs_historical": difference_metrics(
                        (
                            -current.projectedLinear @ returned_last_state
                            - current.projectedNonlinear
                            @ current.nonlinear_function(returned_last_state)
                        ),
                        raw_rhs(returned_last_state),
                    ),
                }
            short_horizons = {}
            for horizon in HORIZONS:
                result = _solve_direct(
                    raw_rhs,
                    current_initial,
                    np.asarray(shared.time),
                    horizon,
                    config,
                )
                short_horizons[str(horizon)] = reduced_integration_diagnostics(
                    result, horizon
                )
            scaled_result = _solve_direct(
                scaled_rhs,
                scaled_initial_result.x,
                np.asarray(shared.time),
                config.time.final_time,
                config,
            )
            scaled_solve = reduced_integration_diagnostics(
                scaled_result, config.time.final_time
            )
            model_reports[model_name] = {
                "lifting": lifting_report,
                "projected_operators": operator_report,
                "initial_coordinate": {
                    "current_direct_ridge": raw_initial_diagnostics,
                    "paper_scaled_diagnostic": scaled_initial_diagnostics,
                },
                "rhs_parity_and_growth": rhs_report,
                "authoritative_current_failed_radau_rerun": full_failure,
                "short_horizon_current_diagnostics": short_horizons,
                "paper_scaled_projected_diagnostic": {
                    "non_authoritative_until_scaling_correction_is_accepted": True,
                    "solve": scaled_solve,
                },
            }
            internal_models[model_name] = {
                "context": context,
                "inferred_case": inferred_case,
                "raw_lift": raw_lift,
                "scaled_lift": scaled_lift,
                "raw_reference_operators": raw_reference_operators,
                "scaled_reference_operators": scaled_reference_operators,
            }
            report["models"] = model_reports
            _write_json(report_path, report)

        print("running inference histories and provenance variants", flush=True)
        variants = (
            ("A_current_corrected", 7501, False),
            ("B_training_count_7500_corrected", 7500, False),
            ("C_7500_historical_derivative_defect", 7500, True),
            ("D_7501_historical_derivative_defect", 7501, True),
        )
        inference_report: dict[str, Any] = {}
        for model_name in MODELS:
            info = internal_models[model_name]
            context = info["context"]
            inferred_case = info["inferred_case"]
            shared_projected_derivative = np.asarray(
                context.model.projectedDerivativeLinear
            )
            corrected_from_coefficients = latent_derivatives(
                latent_coefficients,
                base_config.time.output_spacing,
            )
            latent_absorption_minus_scattering = (
                np.asarray(context.model.projectedAbsorptionLinear)
                - np.asarray(context.model.projectedScatteringLinear)
            )
            model_inference: dict[str, Any] = {
                "shared_derivative_vs_latent_stencil": difference_metrics(
                    shared_projected_derivative, corrected_from_coefficients
                ),
                "semantics": {},
            }
            internal_semantics: dict[str, dict[str, dict[str, Any]]] = {}
            for semantics in ("current_direct", "paper_scaled"):
                model_inference["semantics"][semantics] = {}
                internal_semantics[semantics] = {}
                for name, variant_count, defective in variants:
                    print(
                        f"inference {model_name} {semantics} {name}",
                        flush=True,
                    )
                    public, internal = _inference_variant(
                        name=name,
                        count=variant_count,
                        defective=defective,
                        semantics=semantics,
                        model=model_name,
                        gamma=float(inferred_case.lifting_regularization_gamma),
                        lambda_linear=float(inferred_case.lambda_L),
                        lambda_quadratic=float(inferred_case.lambda_Q),
                        latent_coefficients=latent_coefficients,
                        lifting_coefficients=lifting_coefficients,
                        shared_projected_derivative=shared_projected_derivative,
                        latent_absorption_minus_scattering=latent_absorption_minus_scattering,
                        lifting_basis=lifting_basis,
                        latent_basis=latent_basis,
                        operators=operators,
                        dt=base_config.time.output_spacing,
                        tolerance=float(inferred_case.inference_tolerance),
                        maximum_iterations=int(inferred_case.maximum_iterations),
                    )
                    model_inference["semantics"][semantics][name] = public
                    internal_semantics[semantics][name] = internal
                baseline = internal_semantics[semantics]["A_current_corrected"]
                for name, _variant_count, _defective in variants:
                    internal = internal_semantics[semantics][name]
                    model_inference["semantics"][semantics][name][
                        "operator_difference_from_A"
                    ] = {
                        "linear": difference_metrics(
                            internal["linear_operator"],
                            baseline["linear_operator"],
                        ),
                        "nonlinear": difference_metrics(
                            internal["nonlinear_operator"],
                            baseline["nonlinear_operator"],
                        ),
                        "lifting": difference_metrics(
                            internal["lifting_matrix"],
                            baseline["lifting_matrix"],
                        ),
                    }
            raw_a = internal_semantics["current_direct"]["A_current_corrected"]
            scaled_a = internal_semantics["paper_scaled"]["A_current_corrected"]
            model_inference["paper_scaled_A_vs_current_direct_A"] = {
                "linear_operator": difference_metrics(
                    scaled_a["linear_operator"], raw_a["linear_operator"]
                ),
                "nonlinear_operator": difference_metrics(
                    scaled_a["nonlinear_operator"], raw_a["nonlinear_operator"]
                ),
                "lifting_matrix": difference_metrics(
                    scaled_a["lifting_matrix"], raw_a["lifting_matrix"]
                ),
            }
            raw_public = model_inference["semantics"]["current_direct"][
                "A_current_corrected"
            ]
            raw_residual = (
                shared_projected_derivative
                + latent_absorption_minus_scattering @ latent_coefficients
                + (
                    latent_basis.T
                    @ operators.total_interaction
                    @ (lifting_basis @ info["raw_lift"]["lifting_matrix"])
                    - latent_basis.T
                    @ operators.scattering
                    @ (lifting_basis @ info["raw_lift"]["lifting_matrix"])
                )
                @ info["raw_lift"]["features"]
            )
            current_linear, current_nonlinear, current_diagnostics = (
                NonlinearManifoldReducedModel.nonlinear_inference(
                    latent_coefficients,
                    info["raw_lift"]["features"],
                    raw_residual,
                    ll_A=float(inferred_case.lambda_L),
                    ll_H=float(inferred_case.lambda_Q),
                    tolerance=float(inferred_case.inference_tolerance),
                    max_iterations=int(inferred_case.maximum_iterations),
                    return_diagnostics=True,
                )
            )
            model_inference["current_class_vs_historical_sequence_A"] = {
                "linear_operator": difference_metrics(
                    current_linear, raw_a["linear_operator"]
                ),
                "nonlinear_operator": difference_metrics(
                    current_nonlinear, raw_a["nonlinear_operator"]
                ),
                "iteration_count_current": current_diagnostics["iteration_count"],
                "iteration_count_reference": raw_public["iteration_count"],
                "final_measure_current": current_diagnostics[
                    "final_convergence_measure"
                ],
                "final_measure_reference": raw_public[
                    "final_convergence_measure"
                ],
            }
            inference_report[model_name] = model_inference
            report["inference"] = inference_report
            _write_json(report_path, report)

        report.update(
            {
                "status": "completed",
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "correction_applied": True,
                "correction": {
                    "scope": "publication ROM execution only",
                    "description": (
                        "multiply catalog gamma, lambda_L, and lambda_Q coefficients "
                        "by the validated training snapshot count exactly once at the "
                        "Gram-matrix solves"
                    ),
                    "legacy_general_ROM_semantics_changed": False,
                    "catalog_values_changed": False,
                },
            }
        )
        _write_json(report_path, report)
        print(json.dumps({"status": "completed", "report": str(report_path)}, indent=2))
        return 0
    except Exception as error:
        report.update(
            {
                "status": "failed",
                "finish_time_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
