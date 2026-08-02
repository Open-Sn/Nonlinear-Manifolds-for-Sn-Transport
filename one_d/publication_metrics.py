"""Author-approved publication metrics and timing definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


INSTANTANEOUS_ERROR_DEFINITION = {
    "metric_id": "instantaneous_steady_state_normalized_mass_error",
    "formula": "sqrt((d(t)^T M d(t)) / (psi_inf^T M psi_inf))",
    "difference": "d(t) = psi_FOM(t) - psi_ROM(t)",
    "normalization": "steady_state_mass_norm",
    "mass_convention": "spatial_DG_mass_repeated_by_angle_without_angular_weights",
    "temporal_aggregation": None,
    "status": "fully_specified",
}

RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION = {
    "metric_id": "relative_space_time_l2_error_v1",
    "metric_provenance": "author_approved_repository_definition",
    "formula": (
        "sqrt(trapezoid(d(t)^T M d(t), t) / "
        "trapezoid(psi_FOM(t)^T M psi_FOM(t), t))"
    ),
    "difference": "d(t) = psi_FOM(t) - psi_ROM(t)",
    "temporal_quadrature": "trapezoidal",
    "endpoint_policy": "both_endpoints_included",
    "time_interval": [0.0, 10.0],
    "integrand_power": "squared_M_norm",
    "final_square_root": True,
    "reference_field": "uncentered_transient_fom",
    "mass_convention": "spatial_DG_mass_repeated_by_angle_without_angular_weights",
    "status": "fully_specified",
    "historical_source_status": "not_recovered_from_historical_executable_source",
}

ONLINE_TIMING_DEFINITION = {
    "online_timing_id": "rom_solve_ivp_only_v1",
    "timing_provenance": "author_approved_repository_definition",
    "online_boundary": "wall_clock_inside_reduced_solve_ivp_call_only",
    "speedup_formula": (
        "validated_production_fom_integration_elapsed / "
        "rom_online_integration_elapsed"
    ),
    "speedup_denominator": "rom_online_integration_elapsed",
    "speedup_numerator": "validated_production_fom_integration_elapsed",
    "measurement_policy": {
        "measured_runs_per_case": 1,
        "warmup_runs": 0,
        "repeated_run_average": False,
        "machine_specific": True,
        "scientific_golden_value": False,
    },
    "required_fields": [
        "online_runtime_seconds",
        "offline_runtime_seconds",
        "total_runtime_seconds",
        "speedup_basis",
        "included_stages",
        "excluded_stages",
    ],
    "excluded_from_online": [
        "snapshot loading",
        "steady-state computation",
        "training selection",
        "derivatives",
        "POD/SVD",
        "lifting regression",
        "projected-operator construction",
        "operator inference",
        "nonlinear initial-coordinate optimization",
        "reconstruction",
        "error calculation",
        "artifact writing",
        "plotting",
    ],
    "historical_source_status": "not_recovered_from_historical_executable_source",
    "status": "fully_specified",
}

CONVERGENCE_METRIC_REQUIREMENT = RELATIVE_SPACE_TIME_L2_ERROR_DEFINITION
TIMING_REQUIREMENT = ONLINE_TIMING_DEFINITION

LEGACY_CONVERGENCE_METRIC_REQUIREMENT_V2 = {
    "metric_id": "publication_time_aggregated_relative_error",
    "status": "requires_author_input",
    "missing": [
        "pointwise numerator (M-norm or squared M-norm)",
        "denominator field (transient FOM, centered transient, or steady state)",
        "denominator power (M-norm or squared M-norm)",
        "temporal quadrature rule and endpoint weights",
        "final square-root convention",
        "integration interval ([0,10], [0,7.5], (7.5,10], or another interval)",
    ],
    "prohibited_substitutions": [
        "arithmetic mean",
        "RMS",
        "trapezoidal integral",
        "rectangle-rule integral",
        "mean of instantaneous relative errors",
    ],
}

LEGACY_CONVERGENCE_METRIC_REQUIREMENT_V1 = {
    **LEGACY_CONVERGENCE_METRIC_REQUIREMENT_V2,
    "missing": [
        "numerator definition",
        "denominator definition",
        "whether the M-norm is squared before temporal aggregation",
        "temporal quadrature rule and endpoint treatment",
        "whether a final square root is applied",
        "treatment of training and extrapolation intervals",
    ],
}

LEGACY_TIMING_REQUIREMENT_V1 = {
    "status": "requires_explicit_classification",
    "paper_speedup_scope": "online_only_excluding_offline_costs",
    "required_fields": ONLINE_TIMING_DEFINITION["required_fields"],
    "note": "Exact publication timing boundaries are not established in repository provenance.",
}


class MetricDefinitionUnavailable(RuntimeError):
    """Backward-compatible exception retained for external imports."""


@dataclass(frozen=True)
class PodEnergyCurves:
    eigenvalues: np.ndarray
    retained_energy_fraction: np.ndarray
    unresolved_energy_fraction: np.ndarray
    basis_dimensions: np.ndarray


def instantaneous_error_history(
    fom_state: np.ndarray,
    rom_state: np.ndarray,
    mass_matrix: Any,
    steady_state: np.ndarray,
) -> np.ndarray:
    """Compute the preserved instantaneous error with the existing mass convention."""
    fom = np.asarray(fom_state, dtype=float)
    rom = np.asarray(rom_state, dtype=float)
    steady = np.asarray(steady_state, dtype=float)
    if fom.ndim != 2 or rom.ndim != 2 or fom.shape != rom.shape:
        raise ValueError("FOM and ROM states must be two-dimensional with equal shapes")
    if steady.shape != (fom.shape[0],):
        raise ValueError("steady_state shape must match the state dimension")
    if not np.all(np.isfinite(fom)) or not np.all(np.isfinite(rom)):
        raise ValueError("FOM and ROM states must be finite")
    if not np.all(np.isfinite(steady)):
        raise ValueError("steady_state must be finite")

    denominator = float(steady @ mass_matrix.dot(steady))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("steady-state squared mass norm must be positive and finite")
    difference = fom - rom
    numerator = np.asarray(
        [difference[:, index] @ mass_matrix.dot(difference[:, index]) for index in range(difference.shape[1])],
        dtype=float,
    )
    if np.any(numerator < -1.0e-13) or not np.all(np.isfinite(numerator)):
        raise ValueError("squared mass errors must be nonnegative and finite")
    return np.sqrt(np.maximum(numerator, 0.0) / denominator)


def pod_energy_curves(singular_values: np.ndarray) -> PodEnergyCurves:
    """Convert POD singular values into retained and unresolved energy curves."""
    values = np.asarray(singular_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("singular_values must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("singular_values must be finite and nonnegative")
    eigenvalues = values**2
    total = float(np.sum(eigenvalues))
    if total <= 0.0:
        raise ValueError("POD energy must be positive")
    retained = np.cumsum(eigenvalues) / total
    unresolved = np.maximum(1.0 - retained, 0.0)
    return PodEnergyCurves(
        eigenvalues=eigenvalues,
        retained_energy_fraction=retained,
        unresolved_energy_fraction=unresolved,
        basis_dimensions=np.arange(1, values.size + 1, dtype=int),
    )


def _validate_approved_time(time: np.ndarray, columns: int) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.shape != (columns,):
        raise ValueError("time must have one entry per trajectory column")
    if not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("time must be finite and strictly increasing")
    if not np.isclose(values[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
        values[-1], 10.0, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("relative_space_time_l2_error requires the full [0,10] interval")
    return values


def relative_space_time_l2_error_from_energies(
    error_energy: np.ndarray,
    reference_energy: np.ndarray,
    time: np.ndarray,
) -> float:
    """Integrate approved squared M-norm histories with endpoint trapezoids."""
    numerator_values = np.asarray(error_energy, dtype=float)
    denominator_values = np.asarray(reference_energy, dtype=float)
    if numerator_values.ndim != 1 or denominator_values.shape != numerator_values.shape:
        raise ValueError("error and reference energies must be equal one-dimensional arrays")
    values = _validate_approved_time(time, numerator_values.size)
    if not np.all(np.isfinite(numerator_values)) or not np.all(
        np.isfinite(denominator_values)
    ):
        raise ValueError("energy histories must be finite")
    tolerance = 1.0e-12 * max(
        1.0,
        float(np.max(np.abs(numerator_values))),
        float(np.max(np.abs(denominator_values))),
    )
    if np.any(numerator_values < -tolerance) or np.any(
        denominator_values < -tolerance
    ):
        raise ValueError("squared M-norm histories must be nonnegative")
    numerator = float(np.trapz(np.maximum(numerator_values, 0.0), values))
    denominator = float(np.trapz(np.maximum(denominator_values, 0.0), values))
    if not np.isfinite(numerator) or numerator < 0.0:
        raise ValueError("integrated error energy must be finite and nonnegative")
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("integrated uncentered FOM energy must be positive and finite")
    return float(np.sqrt(numerator / denominator))


def relative_space_time_l2_error(
    fom_state: np.ndarray,
    rom_state: np.ndarray,
    mass_matrix: Any,
    time: np.ndarray,
) -> float:
    """Compute the author-approved full-interval relative space-time error."""
    fom = np.asarray(fom_state, dtype=float)
    rom = np.asarray(rom_state, dtype=float)
    if fom.ndim != 2 or rom.shape != fom.shape:
        raise ValueError("FOM and ROM trajectories must be equal two-dimensional arrays")
    values = _validate_approved_time(time, fom.shape[1])
    if not np.all(np.isfinite(fom)) or not np.all(np.isfinite(rom)):
        raise ValueError("FOM and ROM trajectories must be finite")
    difference = fom - rom
    error_energy = np.sum(difference * mass_matrix.dot(difference), axis=0)
    reference_energy = np.sum(fom * mass_matrix.dot(fom), axis=0)
    return relative_space_time_l2_error_from_energies(
        error_energy,
        reference_energy,
        values,
    )


def publication_convergence_metric(
    fom_state: np.ndarray,
    rom_state: np.ndarray,
    mass_matrix: Any,
    time: np.ndarray,
) -> float:
    """Compatibility name for the approved Figure 4/5 convergence metric."""
    return relative_space_time_l2_error(fom_state, rom_state, mass_matrix, time)


def validate_timing_metadata(timing: dict[str, Any]) -> dict[str, Any]:
    """Validate explicit online/offline timing classification metadata."""
    required = set(TIMING_REQUIREMENT["required_fields"])
    missing = sorted(required.difference(timing))
    if missing:
        raise ValueError("timing metadata is missing: " + ", ".join(missing))
    for field in (
        "online_runtime_seconds",
        "offline_runtime_seconds",
        "total_runtime_seconds",
    ):
        value = timing[field]
        if value is not None and (not np.isfinite(value) or value < 0.0):
            raise ValueError(f"{field} must be null or a finite nonnegative value")
    if not isinstance(timing["included_stages"], list) or not isinstance(
        timing["excluded_stages"], list
    ):
        raise ValueError("included_stages and excluded_stages must be lists")
    if timing["speedup_basis"] is not None and not isinstance(
        timing["speedup_basis"], str
    ):
        raise ValueError("speedup_basis must be null or a string")
    return timing


def publication_metric_definitions() -> dict[str, Any]:
    return {
        "instantaneous_error_history": INSTANTANEOUS_ERROR_DEFINITION,
        "convergence_metric": CONVERGENCE_METRIC_REQUIREMENT,
        "timing": TIMING_REQUIREMENT,
    }


def publication_metric_definitions_compatible(recorded: Any) -> bool:
    """Accept current metadata and immutable pre-approval artifact wording."""
    if not isinstance(recorded, dict):
        return False
    expected = publication_metric_definitions()
    if recorded.get("instantaneous_error_history") != expected["instantaneous_error_history"]:
        return False
    if recorded.get("timing") not in (
        expected["timing"],
        LEGACY_TIMING_REQUIREMENT_V1,
    ):
        return False
    convergence = recorded.get("convergence_metric")
    return convergence in (
        expected["convergence_metric"],
        LEGACY_CONVERGENCE_METRIC_REQUIREMENT_V2,
        LEGACY_CONVERGENCE_METRIC_REQUIREMENT_V1,
    )
