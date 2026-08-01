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


def initialize_rom_context_from_precomputed(
    config: OneDConfig,
    snapshot: np.ndarray,
    *,
    steady_state: np.ndarray,
    time: np.ndarray,
    training_indices: np.ndarray,
    extrapolation_indices: np.ndarray,
    derivatives: np.ndarray,
    basis: np.ndarray,
    singular_values: np.ndarray,
    coefficients: np.ndarray,
    model_name: str | None = None,
    operator_choice: str | None = None,
    problem: ConfiguredProblem | None = None,
    operators: AssembledOperators | None = None,
) -> RomContext:
    """Prepare a ROM from lossless shared POD and derivative artifacts.

    This installs the same model attributes produced by ``compute_time_derivatives``
    and ``compute_pod``.  It intentionally does not recompute either expensive stage.
    """
    model_name = normalize_model_name(model_name or config.rom.embedding_type)
    operator_choice = operator_choice or config.rom.streaming_operators
    if operator_choice not in {"projected", "inferred"}:
        raise ValueError("operator_choice must be 'projected' or 'inferred'")
    if problem is None:
        problem = build_problem(config)
    if operators is None:
        operators = assemble_operators(problem)

    snapshot = np.asarray(snapshot)
    steady_state = np.asarray(steady_state)
    time = np.asarray(time)
    training_indices = np.asarray(training_indices)
    extrapolation_indices = np.asarray(extrapolation_indices)
    derivatives = np.asarray(derivatives)
    basis = np.asarray(basis)
    singular_values = np.asarray(singular_values)
    coefficients = np.asarray(coefficients)
    state_size, output_count = config.expected_snapshot_shape
    training_count = training_indices.size
    total_dimension = config.rom.latent_dimension + config.rom.lifting_dimension
    expected = {
        "snapshot": (snapshot.shape, (state_size, output_count)),
        "steady_state": (steady_state.shape, (state_size,)),
        "time": (time.shape, (output_count,)),
        "derivatives": (derivatives.shape, (state_size, training_count)),
    }
    for name, (received, required) in expected.items():
        if received != required:
            raise ValueError(f"precomputed {name} shape mismatch: expected {required}, received {received}")
    if basis.ndim != 2 or basis.shape[0] != state_size or basis.shape[1] < total_dimension:
        raise ValueError("precomputed POD basis does not cover the requested dimensions")
    if coefficients.ndim != 2 or coefficients.shape[0] < total_dimension or coefficients.shape[1] != training_count:
        raise ValueError("precomputed POD coefficients do not cover the requested dimensions")
    if singular_values.ndim != 1 or singular_values.size < total_dimension:
        raise ValueError("precomputed singular values do not cover the requested dimensions")
    if not np.array_equal(training_indices, np.arange(training_count)):
        raise ValueError("precomputed training indices must be the leading inclusive time interval")
    expected_time = build_time_array(config)
    if not np.array_equal(time, expected_time):
        raise ValueError("precomputed time array does not match the resolved configuration")
    if not np.array_equal(
        extrapolation_indices, np.arange(training_count, output_count)
    ):
        raise ValueError("precomputed extrapolation indices do not follow the training interval")

    legacy_rom = _install_operator_context(operators)
    embedding = None if model_name == "linear" else model_name
    model = legacy_rom.NonlinearManifoldReducedModel(embedding)
    model.solution_path = None
    model.solutionDG1 = snapshot
    model.solutionInf = steady_state
    model.global_training_set = None
    model.global_derivative_set = derivatives
    model.TT = config.time.final_time
    model.dt = config.time.output_spacing
    model.time_steps = time
    model.training_end_time = config.time.training_end_time
    model.training_indices = training_indices
    model.extrapolation_indices = extrapolation_indices
    model.train_size = training_count
    model.n_dofs = state_size
    model.size_R = config.rom.latent_dimension
    model.size_Q = config.rom.lifting_dimension
    model.basis = basis
    model.svd_val = singular_values
    model.coefficients = coefficients
    end = total_dimension
    rank = config.rom.latent_dimension
    model.pod_linear_basis = basis[:, :rank]
    model.pod_linear_coeff = coefficients[:rank, :]
    model.pod_ortho_basis = basis[:, rank:end]
    model.pod_ortho_coeff = coefficients[rank:end, :]
    model.pod_global_basis = basis[:, :end]
    model.pod_global_coeff = coefficients[:end, :]
    return RomContext(
        config=config,
        model_name=model_name,
        operator_choice=operator_choice,
        problem=problem,
        operators=operators,
        snapshot=snapshot,
        steady_state=steady_state,
        time=time,
        training_indices=training_indices,
        extrapolation_indices=extrapolation_indices,
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


def _validated_regularization_scale(regularization_scale: float) -> float:
    scale = float(regularization_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("regularization_scale must be positive and finite")
    return scale


def regularization_diagnostics(
    context: RomContext, *, regularization_scale: float = 1.0
) -> dict[str, Any]:
    """Report the coefficients and actual Gram-matrix ridge terms."""
    scale = _validated_regularization_scale(regularization_scale)
    gamma = float(context.config.rom.lifting_regularization)
    lambda_l = float(context.config.rom.linear_inference_regularization)
    lambda_q = float(
        context.config.rom.quadratic_regularization_for(context.model_name)
    )
    return {
        "coefficient_scale": scale,
        "lifting_gamma_coefficient": gamma,
        "lifting_gram_ridge_actual": gamma * scale,
        "linear_lambda_coefficient": lambda_l,
        "linear_inference_gram_ridge_actual": lambda_l * scale,
        "quadratic_lambda_coefficient": lambda_q,
        "quadratic_inference_gram_ridge_actual": lambda_q * scale,
    }


def construct_nonlinear_lifting(
    context: RomContext, *, regularization_scale: float = 1.0
) -> Any:
    if context.model_name == "linear":
        return None
    scale = _validated_regularization_scale(regularization_scale)
    context.model.compute_nonlinear_embedding(
        lambda_E=context.config.rom.lifting_regularization * scale
    )
    return context.model.pod_nonlinear_basis


def construct_projected_operators(context: RomContext) -> Any:
    context.model.compute_projected_operators()
    return context.model.projectedLinear


def construct_inferred_operators(
    context: RomContext, *, regularization_scale: float = 1.0
) -> Any:
    scale = _validated_regularization_scale(regularization_scale)
    context.model.compute_inferred_operators(
        lambda_A=context.config.rom.linear_inference_regularization * scale,
        lambda_H=context.config.rom.quadratic_regularization_for(
            context.model_name
        )
        * scale,
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


def reconstruct_and_compute_errors_chunked(
    context: RomContext,
    reduced_state: np.ndarray,
    *,
    selected_indices: tuple[int, ...] = (),
    chunk_size: int = 256,
) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Apply the preserved reconstruction and mass-error formulas in chunks."""
    reduced_state = np.asarray(reduced_state)
    if reduced_state.ndim != 2 or reduced_state.shape[1] != context.time.size:
        raise ValueError("reduced_state must have one column per configured output time")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    selected = set(int(index) for index in selected_indices)
    if any(index < 0 or index >= context.time.size for index in selected):
        raise ValueError("selected reconstruction index is out of range")
    denominator = float(
        context.steady_state
        @ context.operators.mass.dot(context.steady_state)
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("steady-state squared mass norm must be positive and finite")
    squared_errors = np.empty(context.time.size, dtype=float)
    selected_fields: dict[int, np.ndarray] = {}
    for start in range(0, context.time.size, chunk_size):
        stop = min(start + chunk_size, context.time.size)
        reconstruction = context.model.reconstruct(reduced_state[:, start:stop])
        difference = reconstruction - context.snapshot[:, start:stop]
        mass_difference = context.operators.mass.dot(difference)
        squared_errors[start:stop] = np.sum(difference * mass_difference, axis=0)
        for index in selected.intersection(range(start, stop)):
            selected_fields[index] = np.asarray(
                reconstruction[:, index - start]
            ).copy()
    if not np.all(np.isfinite(squared_errors)):
        raise RuntimeError("chunked reconstruction produced non-finite mass errors")
    tolerance = 1.0e-13 * max(1.0, float(np.max(np.abs(squared_errors))))
    if np.any(squared_errors < -tolerance):
        raise RuntimeError("chunked reconstruction produced negative squared mass errors")
    return (
        np.sqrt(np.maximum(squared_errors, 0.0) / denominator),
        selected_fields,
    )


def run_selected_rom(
    config: OneDConfig,
    snapshot_path: str,
    *,
    model_name: str | None = None,
    operator_choice: str | None = None,
    regularization_scale: float = 1.0,
) -> RomRunResult:
    """Execute one ROM case.

    ``regularization_scale`` defaults to one for legacy configurations whose
    values already denote actual Gram-matrix ridge terms. Publication callers
    pass their training snapshot count because catalog values are the
    coefficients in the paper's ``coefficient * N_s`` formulas.
    """
    snapshot = load_and_validate_snapshots(config, snapshot_path)
    context = initialize_rom_context(
        config,
        snapshot,
        model_name=model_name,
        operator_choice=operator_choice,
    )
    compute_derivatives(context)
    compute_pod_data(context)
    construct_nonlinear_lifting(
        context, regularization_scale=regularization_scale
    )
    construct_projected_operators(context)
    if context.operator_choice == "inferred":
        construct_inferred_operators(
            context, regularization_scale=regularization_scale
        )
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
        "regularization": regularization_diagnostics(
            context, regularization_scale=regularization_scale
        ),
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
