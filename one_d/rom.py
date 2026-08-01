"""Reusable stages for one explicitly selected one-dimensional ROM case."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sparse

from .config import OneDConfig
from .fom import build_time_array, load_snapshot
from .problem import (
    AssembledOperators,
    ConfiguredProblem,
    assemble_operators,
    build_problem,
    mass_matrix_square_root,
)


def normalize_model_name(model: str) -> str:
    aliases = {
        "linear": "linear",
        "elementwise": "elementwise",
        "element-wise": "elementwise",
        "poly": "elementwise",
        "tensorial": "tensorial",
        "tens": "tensorial",
    }
    try:
        return aliases[model]
    except KeyError:
        raise ValueError(
            "model must be 'linear', 'elementwise', or 'tensorial'"
        ) from None


def partition_time_indices(
    time: np.ndarray, training_end_time: float
) -> tuple[np.ndarray, np.ndarray]:
    """Select the inclusive physical-time training interval."""
    time = np.asarray(time, dtype=float)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be finite and strictly increasing")
    tolerance = max(
        1.0e-12,
        16.0 * np.finfo(float).eps * max(1.0, np.max(np.abs(time))),
    )
    training = np.flatnonzero(time <= float(training_end_time) + tolerance)
    extrapolation = np.flatnonzero(time > float(training_end_time) + tolerance)
    if training.size == 0:
        raise ValueError("training_end_time precedes the first FOM time")
    return training, extrapolation


@dataclass
class RomContext:
    config: OneDConfig
    model_name: str
    operator_choice: str
    problem: ConfiguredProblem
    operators: AssembledOperators
    snapshot: np.ndarray
    steady_state: np.ndarray
    time: np.ndarray
    training_indices: np.ndarray
    extrapolation_indices: np.ndarray
    model: Any


@dataclass(frozen=True)
class RomRunResult:
    model_name: str
    operator_choice: str
    time: np.ndarray
    reduced_state: np.ndarray
    reconstructed_state: np.ndarray
    errors: np.ndarray
    diagnostics: dict[str, Any]


def load_and_validate_snapshots(
    config: OneDConfig, snapshot_path: str
) -> np.ndarray:
    return load_snapshot(snapshot_path, config)


def compute_steady_state(operators: AssembledOperators) -> np.ndarray:
    steady = sparse.linalg.spsolve(operators.system, operators.boundary_source)
    if not np.all(np.isfinite(steady)):
        raise RuntimeError("configured steady-state solve returned non-finite values")
    return np.asarray(steady)


def select_training_interval(
    config: OneDConfig,
    snapshot: np.ndarray,
    steady_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = build_time_array(config)
    training, extrapolation = partition_time_indices(
        time, config.time.training_end_time
    )
    centered_training = snapshot[:, training] - steady_state[:, None]
    return time, training, extrapolation, centered_training


def _install_operator_context(operators: AssembledOperators):
    import Nonlinear_Manifold_ROM as legacy_rom

    legacy_rom.globalMM = operators.mass
    legacy_rom.globalMMsqrt = mass_matrix_square_root(operators)
    legacy_rom.globalAbsorption = operators.total_interaction
    legacy_rom.globalScattering = operators.scattering
    legacy_rom.globalStreaming = operators.streaming
    legacy_rom.globalFF = operators.system
    legacy_rom.globalRB = operators.boundary_source
    return legacy_rom


def initialize_rom_context(
    config: OneDConfig,
    snapshot: np.ndarray,
    *,
    model_name: str | None = None,
    operator_choice: str | None = None,
    problem: ConfiguredProblem | None = None,
    operators: AssembledOperators | None = None,
) -> RomContext:
    """Prepare explicit ROM state without running POD, inference, or integration."""
    model_name = normalize_model_name(model_name or config.rom.embedding_type)
    operator_choice = operator_choice or config.rom.streaming_operators
    if operator_choice not in {"projected", "inferred"}:
        raise ValueError("operator_choice must be 'projected' or 'inferred'")
    if problem is None:
        problem = build_problem(config)
    if operators is None:
        operators = assemble_operators(problem)
    steady = compute_steady_state(operators)
    time, training, extrapolation, centered = select_training_interval(
        config, snapshot, steady
    )
    legacy_rom = _install_operator_context(operators)
    embedding = None if model_name == "linear" else model_name
    model = legacy_rom.NonlinearManifoldReducedModel(embedding)
    model.solution_path = None
    model.solutionDG1 = snapshot
    model.solutionInf = steady
    model.global_training_set = centered
    model.TT = config.time.final_time
    model.dt = config.time.output_spacing
    model.time_steps = time
    model.training_end_time = config.time.training_end_time
    model.training_indices = training
    model.extrapolation_indices = extrapolation
    model.train_size = training.size
    model.n_dofs = snapshot.shape[0]
    return RomContext(
        config=config,
        model_name=model_name,
        operator_choice=operator_choice,
        problem=problem,
        operators=operators,
        snapshot=snapshot,
        steady_state=steady,
        time=time,
        training_indices=training,
        extrapolation_indices=extrapolation,
        model=model,
    )


def compute_derivatives(context: RomContext) -> np.ndarray:
    context.model.compute_time_derivatives()
    return context.model.global_derivative_set


def compute_pod_data(context: RomContext) -> Any:
    context.model.compute_pod(
        size_R=context.config.rom.latent_dimension,
        size_Q=context.config.rom.lifting_dimension,
    )
    return context.model.pod_global_basis


def construct_nonlinear_lifting(context: RomContext) -> Any:
    if context.model_name == "linear":
        return None
    context.model.compute_nonlinear_embedding(
        lambda_E=context.config.rom.lifting_regularization
    )
    return context.model.pod_nonlinear_basis


def construct_projected_operators(context: RomContext) -> Any:
    context.model.compute_projected_operators()
    return context.model.projectedLinear


def construct_inferred_operators(context: RomContext) -> Any:
    context.model.compute_inferred_operators(
        lambda_A=context.config.rom.linear_inference_regularization,
        lambda_H=context.config.rom.quadratic_regularization_for(
            context.model_name
        ),
        tolerance=context.config.rom.nonlinear_inference_tolerance,
        max_iterations=context.config.rom.nonlinear_inference_maximum_iterations,
    )
    return context.model.inferredLinear


def integrate_selected_rom(context: RomContext) -> Any:
    intrusive = context.operator_choice == "projected"
    return context.model.integrate_reduced(
        intrusive=intrusive,
        method=context.config.time.rom_method,
        atol=context.config.time.rom_absolute_tolerance,
        rtol=context.config.time.rom_relative_tolerance,
        initial_time=context.config.time.initial_time,
    )


def reconstruct_full_states(
    context: RomContext, reduced_state: np.ndarray
) -> np.ndarray:
    return context.model.reconstruct(reduced_state)


def compute_error_metrics(
    context: RomContext, reconstruction: np.ndarray
) -> np.ndarray:
    return context.model.compute_errors(reconstruction)


def run_selected_rom(
    config: OneDConfig,
    snapshot_path: str,
    *,
    model_name: str | None = None,
    operator_choice: str | None = None,
) -> RomRunResult:
    """Execute one selected ROM case and return states, errors, and diagnostics."""
    snapshot = load_and_validate_snapshots(config, snapshot_path)
    context = initialize_rom_context(
        config,
        snapshot,
        model_name=model_name,
        operator_choice=operator_choice,
    )
    compute_derivatives(context)
    compute_pod_data(context)
    construct_nonlinear_lifting(context)
    construct_projected_operators(context)
    if context.operator_choice == "inferred":
        construct_inferred_operators(context)
    context.model.compute_initial_conditions()
    integration = integrate_selected_rom(context)
    reconstruction = reconstruct_full_states(context, integration.y)
    errors = compute_error_metrics(context, reconstruction)
    diagnostics = {
        "solver_success": bool(integration.success),
        "solver_message": str(integration.message),
        "inference": context.model.inference_diagnostics,
        "training_snapshot_count": int(context.training_indices.size),
        "extrapolation_snapshot_count": int(context.extrapolation_indices.size),
    }
    return RomRunResult(
        model_name=context.model_name,
        operator_choice=context.operator_choice,
        time=context.time,
        reduced_state=np.asarray(integration.y),
        reconstructed_state=np.asarray(reconstruction),
        errors=np.asarray(errors),
        diagnostics=diagnostics,
    )
