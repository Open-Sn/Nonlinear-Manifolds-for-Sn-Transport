"""Isolated historical-formula references for Phase-5 nonlinear diagnostics.

These helpers reproduce the relevant formulas in commit ``bff6910`` without
loading a snapshot, computing an SVD, or changing the production workflow.
"""

from __future__ import annotations

import time
from typing import Any, Iterable

import numpy as np
import scipy.optimize


def historical_quadratic_features(coefficients: np.ndarray, model: str) -> np.ndarray:
    """Evaluate the pre-refactor ``poly`` or ``tens`` feature ordering."""
    coefficients = np.asarray(coefficients)
    if model in {"elementwise", "poly"}:
        return np.power(coefficients, 2)
    if model in {"tensorial", "tens"}:
        indices = np.triu_indices(coefficients.shape[0])
        products = np.einsum("i...,j...->ij...", coefficients, coefficients)
        return products[indices[0], indices[1], ...]
    raise ValueError("model must be elementwise/poly or tensorial/tens")


def historical_lifting(
    latent_coefficients: np.ndarray,
    lifting_coefficients: np.ndarray,
    ridge: float,
    model: str,
) -> dict[str, np.ndarray]:
    """Reproduce the historical nonlinear lifting regression exactly."""
    latent = np.asarray(latent_coefficients)
    lifting = np.asarray(lifting_coefficients)
    features = historical_quadratic_features(latent, model)
    gram = features @ features.T + float(ridge) * np.eye(features.shape[0])
    solved = np.linalg.solve(gram, features)
    lift = solved.dot(lifting.T).T
    return {
        "features": features,
        "gram": gram,
        "regression_rhs": lifting @ features.T,
        "lifting_matrix": lift,
        "lifted_coefficients": lift @ features,
    }


def historical_projected_operators(
    latent_basis: np.ndarray,
    nonlinear_basis: np.ndarray,
    *,
    streaming: Any,
    absorption: Any,
    scattering: Any,
) -> dict[str, np.ndarray]:
    """Reproduce the pre-refactor ``G + A - B`` projections."""
    latent = np.asarray(latent_basis)
    nonlinear = np.asarray(nonlinear_basis)
    values = {
        "G_r": latent.T @ streaming @ latent,
        "A_r": latent.T @ absorption @ latent,
        "B_r": latent.T @ scattering @ latent,
        "G_lifted": latent.T @ streaming @ nonlinear,
        "A_lifted": latent.T @ absorption @ nonlinear,
        "B_lifted": latent.T @ scattering @ nonlinear,
    }
    values["combined_linear"] = values["G_r"] + values["A_r"] - values["B_r"]
    values["combined_lifted"] = (
        values["G_lifted"] + values["A_lifted"] - values["B_lifted"]
    )
    return values


def historical_initial_coordinate(
    latent_initial: np.ndarray,
    global_initial: np.ndarray,
    lifting_matrix: np.ndarray,
    model: str,
) -> tuple[Any, dict[str, float]]:
    """Run the historical Nelder-Mead initial-coordinate fit with diagnostics."""
    latent_initial = np.asarray(latent_initial)
    global_initial = np.asarray(global_initial)
    lifting_matrix = np.asarray(lifting_matrix)

    def objective(coordinate: np.ndarray) -> float:
        lifted = lifting_matrix @ historical_quadratic_features(coordinate, model)
        return float(np.linalg.norm(global_initial - np.concatenate((coordinate, lifted))))

    initial_objective = objective(latent_initial)
    result = scipy.optimize.minimize(
        objective,
        latent_initial,
        method="Nelder-Mead",
        tol=1.0e-7,
        options={"maxiter": 10000},
    )
    diagnostics = {
        "initial_objective": initial_objective,
        "final_objective": float(result.fun),
        "coordinate_norm": float(np.linalg.norm(result.x)),
        "lifted_coefficient_norm": float(
            np.linalg.norm(
                lifting_matrix @ historical_quadratic_features(result.x, model)
            )
        ),
    }
    return result, diagnostics


def historical_nonlinear_rhs(
    coordinate: np.ndarray,
    linear_operator: np.ndarray,
    nonlinear_operator: np.ndarray,
    model: str,
) -> np.ndarray:
    """Evaluate the historical reduced nonlinear right-hand side."""
    coordinate = np.asarray(coordinate)
    return -np.asarray(linear_operator) @ coordinate - np.asarray(
        nonlinear_operator
    ) @ historical_quadratic_features(coordinate, model)


def historical_alternating_inference(
    latent_coefficients: np.ndarray,
    nonlinear_coefficients: np.ndarray,
    residual: np.ndarray,
    *,
    ridge_linear: float,
    ridge_nonlinear: float,
    tolerance: float,
    maximum_iterations: int,
    checkpoints: Iterable[int] = (),
) -> dict[str, Any]:
    """Reproduce historical alternating updates and retain sparse diagnostics."""
    latent = np.asarray(latent_coefficients)
    nonlinear = np.asarray(nonlinear_coefficients)
    residual = np.asarray(residual)
    rhs_linear = np.linalg.solve(
        latent @ latent.T + float(ridge_linear) * np.eye(latent.shape[0]),
        latent,
    ).T
    rhs_nonlinear = np.linalg.solve(
        nonlinear @ nonlinear.T
        + float(ridge_nonlinear) * np.eye(nonlinear.shape[0]),
        nonlinear,
    ).T
    simple_linear = -residual @ rhs_linear
    simple_nonlinear = -residual @ rhs_nonlinear
    nonlinear_linear = -nonlinear @ rhs_linear
    linear_nonlinear = -latent @ rhs_nonlinear
    linear_operator = np.zeros((latent.shape[0], latent.shape[0]))
    nonlinear_operator = np.zeros((latent.shape[0], nonlinear.shape[0]))
    requested = set(int(value) for value in checkpoints)
    history: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    initial_residual = residual.copy()
    history["0"] = {
        "convergence_measure": None,
        "linear_update_norm": None,
        "nonlinear_update_norm": None,
        "linear_operator_norm": 0.0,
        "nonlinear_operator_norm": 0.0,
        "regression_residual_norm": float(np.linalg.norm(initial_residual)),
        "finite": bool(np.all(np.isfinite(initial_residual))),
        "elapsed_seconds": 0.0,
    }
    converged = False
    termination_reason = "maximum_iterations"
    convergence_measure = float("inf")
    iteration_count = 0
    for iteration_count in range(1, int(maximum_iterations) + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            new_linear = simple_linear + nonlinear_operator @ nonlinear_linear
            new_nonlinear = simple_nonlinear + new_linear @ linear_nonlinear
        finite = bool(
            np.all(np.isfinite(new_linear)) and np.all(np.isfinite(new_nonlinear))
        )
        if not finite:
            linear_operator, nonlinear_operator = new_linear, new_nonlinear
            termination_reason = "nonfinite_iterate"
            break
        linear_update = float(np.linalg.norm(new_linear - linear_operator))
        nonlinear_update = float(np.linalg.norm(new_nonlinear - nonlinear_operator))
        linear_norm = float(np.linalg.norm(new_linear))
        nonlinear_norm = float(np.linalg.norm(new_nonlinear))
        error_linear = linear_update / (linear_norm + np.finfo(float).eps)
        error_nonlinear = nonlinear_update / (nonlinear_norm + np.finfo(float).eps)
        convergence_measure = float(
            np.sqrt(error_linear**2 + error_nonlinear**2)
        )
        if not np.isfinite(convergence_measure):
            linear_operator, nonlinear_operator = new_linear, new_nonlinear
            termination_reason = "nonfinite_convergence_measure"
            break
        linear_operator, nonlinear_operator = new_linear.copy(), new_nonlinear.copy()
        if iteration_count in requested:
            regression_residual = (
                residual
                + linear_operator @ latent
                + nonlinear_operator @ nonlinear
            )
            history[str(iteration_count)] = {
                "convergence_measure": convergence_measure,
                "linear_update_norm": linear_update,
                "nonlinear_update_norm": nonlinear_update,
                "linear_operator_norm": linear_norm,
                "nonlinear_operator_norm": nonlinear_norm,
                "regression_residual_norm": float(
                    np.linalg.norm(regression_residual)
                ),
                "finite": bool(np.all(np.isfinite(regression_residual))),
                "elapsed_seconds": time.perf_counter() - started,
            }
        if convergence_measure < float(tolerance):
            converged = True
            termination_reason = "converged"
            break
    if str(iteration_count) not in history:
        regression_residual = (
            residual + linear_operator @ latent + nonlinear_operator @ nonlinear
        )
        history[str(iteration_count)] = {
            "convergence_measure": convergence_measure,
            "linear_update_norm": None,
            "nonlinear_update_norm": None,
            "linear_operator_norm": float(np.linalg.norm(linear_operator)),
            "nonlinear_operator_norm": float(np.linalg.norm(nonlinear_operator)),
            "regression_residual_norm": float(np.linalg.norm(regression_residual)),
            "finite": bool(np.all(np.isfinite(regression_residual))),
            "elapsed_seconds": time.perf_counter() - started,
        }
    return {
        "linear_operator": linear_operator,
        "nonlinear_operator": nonlinear_operator,
        "rhs_linear": rhs_linear,
        "rhs_nonlinear": rhs_nonlinear,
        "converged": converged,
        "iteration_count": iteration_count,
        "final_convergence_measure": convergence_measure,
        "termination_reason": termination_reason,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }


def latent_derivatives(
    latent_coefficients: np.ndarray,
    dt: float,
    *,
    historical_defect: bool = False,
) -> np.ndarray:
    """Apply corrected or isolated pre-fix eighth-order stencils in latent space."""
    values = np.asarray(latent_coefficients)
    count = values.shape[1]
    if count < 13:
        raise ValueError("at least thirteen snapshots are required")
    central = np.array(
        [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5, -1 / 5, 4 / 105, -1 / 280]
    ) / float(dt)
    forward = np.array(
        [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5, -14 / 3, 8 / 7, -1 / 8]
    ) / float(dt)
    backward = np.array(
        [1 / 8, -8 / 7, 14 / 3, -56 / 5, 35 / 2, -56 / 3, 14, -8, 761 / 280]
    ) / float(dt)
    derivatives = np.zeros_like(values)
    for index in range(4):
        derivatives[:, index] = values[:, index : index + 9] @ forward
        destination = count - 4 + index
        if historical_defect:
            window = values[:, count - 13 : count - 4]
        else:
            window = values[:, destination - 8 : destination + 1]
        derivatives[:, destination] = window @ backward
    for index in range(count - 8):
        derivatives[:, index + 4] = values[:, index : index + 9] @ central
    return derivatives


def difference_metrics(current: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return absolute and relative Frobenius parity metrics."""
    current = np.asarray(current)
    reference = np.asarray(reference)
    difference = current - reference
    denominator = max(float(np.linalg.norm(reference)), np.finfo(float).eps)
    return {
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "relative_frobenius_difference": float(np.linalg.norm(difference) / denominator),
    }
