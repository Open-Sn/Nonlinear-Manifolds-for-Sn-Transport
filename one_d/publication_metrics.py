"""Publication metric definitions without unresolved scientific guesses."""

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

CONVERGENCE_METRIC_REQUIREMENT = {
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

LEGACY_CONVERGENCE_METRIC_MISSING_V1 = [
    "numerator definition",
    "denominator definition",
    "whether the M-norm is squared before temporal aggregation",
    "temporal quadrature rule and endpoint treatment",
    "whether a final square root is applied",
    "treatment of training and extrapolation intervals",
]

TIMING_REQUIREMENT = {
    "status": "requires_explicit_classification",
    "paper_speedup_scope": "online_only_excluding_offline_costs",
    "required_fields": [
        "online_runtime_seconds",
        "offline_runtime_seconds",
        "total_runtime_seconds",
        "speedup_basis",
        "included_stages",
        "excluded_stages",
    ],
    "note": "Exact publication timing boundaries are not established in repository provenance.",
}


class MetricDefinitionUnavailable(RuntimeError):
    """Raised when a publication metric lacks an author-approved definition."""


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


def publication_convergence_metric(*_args: Any, **_kwargs: Any) -> np.ndarray:
    """Refuse to invent the unresolved Figure 4/5 time-aggregated metric."""
    raise MetricDefinitionUnavailable(
        "the publication time-aggregated convergence metric requires author input; "
        "repository provenance does not establish the pointwise numerator "
        "(M-norm or squared M-norm), denominator field and power (transient FOM, "
        "centered transient, or steady state; norm or squared norm), temporal "
        "quadrature and endpoint weights, final square-root convention, or "
        "integration interval ([0,10], [0,7.5], (7.5,10], or another interval)"
    )


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
    """Accept current metadata and the equivalent pre-Phase-6 wording."""
    if not isinstance(recorded, dict):
        return False
    expected = publication_metric_definitions()
    if recorded.get("instantaneous_error_history") != expected["instantaneous_error_history"]:
        return False
    if recorded.get("timing") != expected["timing"]:
        return False
    convergence = recorded.get("convergence_metric")
    current = expected["convergence_metric"]
    if not isinstance(convergence, dict):
        return False
    for key in ("metric_id", "status", "prohibited_substitutions"):
        if convergence.get(key) != current[key]:
            return False
    missing = convergence.get("missing")
    if not isinstance(missing, list):
        return False
    return tuple(missing) in {
        tuple(current["missing"]),
        tuple(LEGACY_CONVERGENCE_METRIC_MISSING_V1),
    }
