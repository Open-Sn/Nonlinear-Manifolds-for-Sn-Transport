"""Reusable full-order stages for configured one-dimensional calculations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import OneDConfig
from .problem import (
    AssembledOperators,
    ConfiguredProblem,
    assemble_operators,
    build_problem,
    construct_initial_condition,
)


@dataclass(frozen=True)
class FomSolution:
    config: OneDConfig
    problem: ConfiguredProblem
    operators: AssembledOperators
    time: np.ndarray
    initial_state: np.ndarray
    solver_result: Any

    @property
    def state(self) -> np.ndarray:
        return np.asarray(self.solver_result.y)


@dataclass(frozen=True)
class SnapshotInspection:
    path: str
    exists: bool
    file_size_bytes: int | None
    shape: tuple[int, ...] | None
    dtype: str | None
    finite: bool | None
    expected_shape: tuple[int, int]
    compatible: bool
    compatibility_errors: tuple[str, ...]
    sha256: str | None
    expected_raw_size_bytes: int
    time_count: int
    initial_time: float
    final_time: float
    output_spacing: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_time_array(config: OneDConfig) -> np.ndarray:
    """Return the validated inclusive configured output grid."""
    return config.time.initial_time + np.arange(
        config.time.output_count, dtype=float
    ) * config.time.output_spacing


def validate_fom_solution(
    result: Any, time: np.ndarray, expected_state_size: int
) -> Any:
    """Validate solver completion, times, shape, and finiteness before use."""
    requested = np.asarray(time, dtype=float)
    returned = np.asarray(getattr(result, "t", np.array([])), dtype=float)
    if not bool(getattr(result, "success", False)):
        message = getattr(result, "message", "no solver message")
        raise RuntimeError(f"configured FOM solve failed: {message}")
    if returned.shape != requested.shape or not np.allclose(
        returned, requested, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError("configured FOM solve returned an inconsistent time grid")
    state = np.asarray(getattr(result, "y", np.array([])))
    expected_shape = (int(expected_state_size), requested.size)
    if state.shape != expected_shape:
        raise RuntimeError(
            f"configured FOM solve expected shape {expected_shape}, received {state.shape}"
        )
    if not np.all(np.isfinite(state)):
        raise RuntimeError("configured FOM solve returned non-finite values")
    return result


def solve_fom(
    config: OneDConfig,
    problem: ConfiguredProblem | None = None,
    operators: AssembledOperators | None = None,
) -> FomSolution:
    """Assemble as needed and execute one explicitly requested FOM solve."""
    import scipy as sp

    if problem is None:
        problem = build_problem(config)
    if operators is None:
        operators = assemble_operators(problem)
    time = build_time_array(config)
    initial_state = construct_initial_condition(problem)
    velocity = config.problem.particle_velocity
    rhs = lambda current_time, state: velocity * operators.inverse_mass.dot(
        operators.boundary_source - operators.system.dot(state)
    )
    result = sp.integrate.solve_ivp(
        fun=rhs,
        t_span=(config.time.initial_time, config.time.final_time),
        y0=initial_state,
        method=config.time.fom_method,
        atol=config.time.fom_absolute_tolerance,
        rtol=config.time.fom_relative_tolerance,
        t_eval=time,
    )
    validate_fom_solution(result, time, config.problem.phase_space_dofs)
    return FomSolution(
        config=config,
        problem=problem,
        operators=operators,
        time=time,
        initial_state=initial_state,
        solver_result=result,
    )


def validate_snapshot_array(array: np.ndarray, config: OneDConfig) -> np.ndarray:
    """Validate a phase-space-by-time snapshot array against configuration."""
    snapshots = np.asarray(array)
    if snapshots.ndim != 2:
        raise ValueError(
            f"snapshot must have rank 2; received rank {snapshots.ndim}"
        )
    if snapshots.shape != config.expected_snapshot_shape:
        raise ValueError(
            f"snapshot shape mismatch: expected {config.expected_snapshot_shape}, "
            f"received {snapshots.shape}"
        )
    if not np.all(np.isfinite(snapshots)):
        raise ValueError("snapshot must contain only finite values")
    return snapshots


def save_snapshot(
    path: str | Path,
    state: np.ndarray,
    config: OneDConfig,
    *,
    overwrite: bool = False,
) -> Path:
    """Validate and save a snapshot with explicit overwrite protection."""
    path = Path(path)
    validated = validate_snapshot_array(state, config)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing snapshot: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, validated)
    return path


def load_snapshot(path: str | Path, config: OneDConfig) -> np.ndarray:
    """Load an existing snapshot and reject incompatible or non-finite data."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"snapshot does not exist: {path}")
    return validate_snapshot_array(np.load(path, allow_pickle=False), config)


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_snapshot(
    path: str | Path, config: OneDConfig, *, include_sha256: bool = False
) -> SnapshotInspection:
    """Inspect a snapshot without modifying it or assembling operators."""
    path = Path(path)
    expected = config.expected_snapshot_shape
    if not path.is_file():
        return SnapshotInspection(
            path=str(path),
            exists=False,
            file_size_bytes=None,
            shape=None,
            dtype=None,
            finite=None,
            expected_shape=expected,
            compatible=False,
            compatibility_errors=("snapshot does not exist",),
            sha256=None,
            expected_raw_size_bytes=config.expected_snapshot_bytes_float64,
            time_count=config.time.output_count,
            initial_time=config.time.initial_time,
            final_time=config.time.final_time,
            output_spacing=config.time.output_spacing,
        )

    errors: list[str] = []
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    finite: bool | None = None
    try:
        snapshots = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = tuple(snapshots.shape)
        dtype = str(snapshots.dtype)
        if snapshots.ndim != 2:
            errors.append(f"expected rank 2, received rank {snapshots.ndim}")
        if snapshots.shape != expected:
            errors.append(f"expected shape {expected}, received {snapshots.shape}")
        try:
            finite = bool(np.all(np.isfinite(snapshots)))
        except TypeError:
            finite = False
        if not finite:
            errors.append("snapshot contains non-finite or nonnumeric values")
    except (OSError, ValueError) as error:
        errors.append(f"unable to load NumPy snapshot: {error}")

    return SnapshotInspection(
        path=str(path),
        exists=True,
        file_size_bytes=path.stat().st_size,
        shape=shape,
        dtype=dtype,
        finite=finite,
        expected_shape=expected,
        compatible=not errors,
        compatibility_errors=tuple(errors),
        sha256=_sha256_file(path) if include_sha256 else None,
        expected_raw_size_bytes=config.expected_snapshot_bytes_float64,
        time_count=config.time.output_count,
        initial_time=config.time.initial_time,
        final_time=config.time.final_time,
        output_spacing=config.time.output_spacing,
    )
