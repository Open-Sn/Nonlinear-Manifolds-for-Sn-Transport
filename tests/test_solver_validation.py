from types import SimpleNamespace

import numpy as np
import pytest

import Nonlinear_Manifold_ROM as rom
import Transport_Driver_Benchmark_1D as driver
from Nonlinear_Manifold_ROM import NonlinearManifoldReducedModel
from Transport_Driver_Benchmark_1D import validate_solve_ivp_result


def _result(times, solution, success=True, message="ok"):
    return SimpleNamespace(
        success=success,
        message=message,
        t=np.asarray(times, dtype=float),
        y=np.asarray(solution, dtype=float),
    )


def test_valid_solver_result_is_returned_unchanged():
    times = np.array([0.0, 0.1, 0.2])
    result = _result(times, np.zeros((2, 3)))
    assert validate_solve_ivp_result(result, times, 2, "test solve") is result


def test_solver_result_validation_checks_requested_final_time():
    times = np.array([0.0, 0.1, 0.2])
    result = _result(times, np.zeros((2, 3)))
    with pytest.raises(RuntimeError, match="time-grid mismatch at final time"):
        validate_solve_ivp_result(
            result,
            times,
            2,
            "test solve",
            expected_final_time=0.3,
        )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_result([0.0], np.zeros((2, 1)), False, "failed"), "solver failure"),
        (_result([0.0, 0.1], np.zeros((2, 2))), "incomplete output"),
        (
            _result([0.0, 0.11, 0.2], np.zeros((2, 3))),
            "time-grid mismatch",
        ),
        (_result([0.0, 0.1, 0.2], np.zeros((3, 3))), "shape mismatch"),
        (
            _result([0.0, 0.1, 0.2], [[0.0, np.nan, 0.0], [0.0, 0.0, 0.0]]),
            "non-finite values",
        ),
    ],
)
def test_solver_result_validation_failure_modes(result, message):
    with pytest.raises(RuntimeError, match=message):
        validate_solve_ivp_result(
            result, np.array([0.0, 0.1, 0.2]), 2, "artificial solve"
        )


def test_full_order_wrapper_rejects_failed_solve(monkeypatch):
    monkeypatch.setattr(driver, "initialize_production_problem", lambda: {})
    monkeypatch.setattr(
        driver.sp.integrate,
        "solve_ivp",
        lambda **kwargs: _result([], np.empty((2, 0)), False, "artificial failure"),
    )
    with pytest.raises(RuntimeError, match="full-order transport solve: solver failure"):
        driver.solve_transport(
            np.zeros(2),
            0.1,
            n_output_times=2,
            evaluation_times=np.array([0.0, 0.1]),
        )


def test_reduced_order_wrapper_rejects_incomplete_solve(monkeypatch):
    model = NonlinearManifoldReducedModel(None)
    model.initial_condition = np.array([1.0])
    model.projectedLinear = np.array([[1.0]])
    model.TT = 0.1
    model.time_steps = np.array([0.0, 0.1])
    monkeypatch.setattr(
        rom.sp.integrate,
        "solve_ivp",
        lambda **kwargs: _result([0.0], np.ones((1, 1))),
    )

    with pytest.raises(RuntimeError, match="reduced-order model solve: incomplete output"):
        model.solve(intrusive=True)
