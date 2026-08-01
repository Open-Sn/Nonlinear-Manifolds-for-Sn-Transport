import numpy as np
import pytest
import scipy.sparse as sparse

import Nonlinear_Manifold_ROM as rom
from Nonlinear_Manifold_ROM import (
    NonlinearManifoldReducedModel,
    partition_time_indices,
)
from Transport_Driver_Benchmark_1D import make_uniform_time_grid


def test_eighth_order_derivatives_are_exact_for_monomials_zero_through_eight():
    dt = 0.05
    time_steps = np.arange(25, dtype=float) * dt
    snapshots = np.vstack([time_steps**degree for degree in range(9)])
    exact = np.vstack(
        [
            np.zeros_like(time_steps)
            if degree == 0
            else degree * time_steps ** (degree - 1)
            for degree in range(9)
        ]
    )

    model = NonlinearManifoldReducedModel(None)
    model.dt = dt
    model.train_size = time_steps.size
    model.global_training_set = snapshots
    model.compute_time_derivatives()

    # The scale-aware combination accommodates cancellation in the one-sided
    # degree-eight formulas while remaining far below scientific tolerances.
    np.testing.assert_allclose(
        model.global_derivative_set, exact, rtol=5e-12, atol=5e-12
    )


def test_backward_derivative_regression_uses_four_distinct_stencils():
    dt = 0.1
    time_steps = np.arange(20, dtype=float) * dt
    model = NonlinearManifoldReducedModel(None)
    model.dt = dt
    model.train_size = time_steps.size
    model.global_training_set = (time_steps**2)[None, :]
    model.compute_time_derivatives()

    np.testing.assert_allclose(
        model.global_derivative_set[0, -4:],
        2.0 * time_steps[-4:],
        rtol=2e-12,
        atol=2e-12,
    )
    assert np.unique(np.round(model.global_derivative_set[0, -4:], 12)).size == 4


def test_derivative_requires_nine_available_training_snapshots():
    model = NonlinearManifoldReducedModel(None)
    model.dt = 0.1
    model.train_size = 8
    model.global_training_set = np.zeros((1, 8))
    try:
        model.compute_time_derivatives()
    except ValueError as error:
        assert "at least nine" in str(error)
    else:
        raise AssertionError("expected a ValueError for fewer than nine snapshots")


def test_production_time_grid_and_inclusive_training_partition():
    time_steps = make_uniform_time_grid(10.0, 0.001)
    training, extrapolation = partition_time_indices(time_steps, 7.5)

    assert time_steps.size == 10001
    assert time_steps[0] == 0.0
    assert time_steps[-1] == 10.0
    np.testing.assert_allclose(np.diff(time_steps), 0.001, rtol=0.0, atol=2e-15)
    assert training.size == 7501
    assert extrapolation.size == 2500
    assert time_steps[training[0]] == 0.0
    assert time_steps[training[-1]] == 7.5
    assert time_steps[extrapolation[0]] == 7.501
    assert time_steps[extrapolation[-1]] == 10.0
    assert np.intersect1d(training, extrapolation).size == 0
    np.testing.assert_array_equal(
        np.sort(np.concatenate((training, extrapolation))),
        np.arange(time_steps.size),
    )


def test_load_training_data_uses_exact_fom_times_and_partition(
    monkeypatch, tmp_path
):
    time_steps = make_uniform_time_grid(10.0, 0.001)
    snapshot_path = tmp_path / "synthetic_snapshots.npy"
    np.save(snapshot_path, np.zeros((2, time_steps.size)))

    monkeypatch.setattr(rom, "_ensure_production_context", lambda: None)
    monkeypatch.setattr(rom, "globalFF", sparse.eye(2, format="csc"))
    monkeypatch.setattr(rom, "globalRB", np.zeros(2))
    model = NonlinearManifoldReducedModel(None)
    model.load_training_data(
        solution_path=snapshot_path,
        train_fraction=0.75,
        TT=10.0,
        dt=0.001,
        evaluation_times=time_steps,
    )

    np.testing.assert_array_equal(model.time_steps, time_steps)
    assert model.train_size == 7501
    assert model.global_training_set.shape == (2, 7501)
    assert model.time_steps[model.training_indices[-1]] == 7.5
    assert model.time_steps[model.extrapolation_indices[0]] == 7.501


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("rank", "rank 2"),
        ("rows", "phase-space rows"),
        ("columns", "time columns: expected 10001"),
        ("nonfinite", "only finite values"),
    ],
)
def test_canonical_snapshot_is_validated_before_use(
    monkeypatch, invalid_kind, message
):
    if invalid_kind == "rank":
        snapshots = np.zeros(10001)
    elif invalid_kind == "rows":
        snapshots = np.zeros((1, 10001))
    elif invalid_kind == "columns":
        snapshots = np.zeros((2, 10000))
    else:
        snapshots = np.zeros((2, 10001))
        snapshots[0, 0] = np.nan

    monkeypatch.setattr(rom, "_ensure_production_context", lambda: None)
    monkeypatch.setattr(rom, "globalFF", sparse.eye(2, format="csc"))
    monkeypatch.setattr(rom, "globalRB", np.zeros(2))
    monkeypatch.setattr(rom.np, "load", lambda path: snapshots)

    model = NonlinearManifoldReducedModel(None)
    with pytest.raises(ValueError, match=message):
        model.load_training_data(solution_path=rom.SOLUTION_PATH)
