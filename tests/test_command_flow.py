from types import SimpleNamespace

import numpy as np
import pytest

import Nonlinear_Manifold_ROM as rom
import Transport_Driver_Benchmark_1D as driver


def test_canonical_production_snapshot_metadata():
    assert driver.SOLUTION_PATH.endswith(
        "Nt10001_Nx750_continuous_bis.npy"
    )
    assert driver.SOLUTION_PATH == rom.SOLUTION_PATH
    assert driver.NN == 10001
    assert driver.PRODUCTION_TIME_STEPS.shape == (10001,)
    assert driver.globalFF.shape == (6000, 6000)


def test_driver_main_generates_missing_snapshot_in_original_sequence(monkeypatch):
    events = []
    original_initialize = driver.initialize_production_problem

    def initialize():
        events.append("initialize")
        return original_initialize()

    def exists(path):
        events.append(("exists", path))
        return False

    def initial_condition(coordinates):
        events.append(("initial_condition", coordinates is driver.xx))
        return np.array([42.0])

    def solve_transport(**kwargs):
        events.append(("solve", kwargs))
        return SimpleNamespace(y=np.full((1, driver.NN), 7.0))

    def save(path, values):
        events.append(("save", path, values.copy()))

    monkeypatch.setattr(driver, "initialize_production_problem", initialize)
    monkeypatch.setattr(driver.os.path, "exists", exists)
    monkeypatch.setattr(driver, "make_production_initial_condition", initial_condition)
    monkeypatch.setattr(driver, "solve_transport", solve_transport)
    monkeypatch.setattr(driver.np, "save", save)

    driver.main()

    assert events[0] == "initialize"
    assert events[1] == ("exists", driver.SOLUTION_PATH)
    assert events[2] == ("initial_condition", True)
    assert events[3][0] == "solve"
    assert events[4][0:2] == ("save", driver.SOLUTION_PATH)
    assert events[3][1]["Psi0"].tolist() == [42.0]
    np.testing.assert_array_equal(
        events[3][1]["psi_bc_func"](0.0), [0.0, 0.0, 0.0, 1.0]
    )
    np.testing.assert_array_equal(events[3][1]["qext_func"](0.0), [0.0, 0.0, 0.0])
    assert events[3][1]["evaluation_times"] is driver.PRODUCTION_TIME_STEPS
    assert not driver.os.path.isabs(driver.SOLUTION_PATH)
    assert (
        driver.SOLUTION_PATH
        == "solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy"
    )
    assert events[4][2].shape == (1, 10001)


def test_driver_main_skips_solve_when_snapshot_exists(monkeypatch):
    def exists(path):
        assert path == "solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy"
        return True

    monkeypatch.setattr(driver.os.path, "exists", exists)
    monkeypatch.setattr(
        driver,
        "solve_transport",
        lambda **kwargs: pytest.fail("existing snapshot must skip FOM solve"),
    )
    monkeypatch.setattr(
        driver.np,
        "save",
        lambda *args, **kwargs: pytest.fail("existing snapshot must not be overwritten"),
    )

    driver.main()


@pytest.mark.parametrize(("snapshot_exists", "expected_fom_calls"), [(False, 1), (True, 0)])
def test_rom_main_generates_fom_only_when_snapshot_is_missing(
    monkeypatch, snapshot_exists, expected_fom_calls
):
    calls = {"fom": 0, "load": []}

    class StopAfterLoad(Exception):
        pass

    class FakeModel:
        def __init__(self, nonlinear_embedding_type=None):
            pass

        def load_training_data(self, **kwargs):
            calls["load"].append(kwargs)
            raise StopAfterLoad

    def fom_main():
        calls["fom"] += 1

    def exists(path):
        assert path == driver.SOLUTION_PATH == rom.SOLUTION_PATH
        return snapshot_exists

    monkeypatch.setattr(rom.os.path, "exists", exists)
    monkeypatch.setattr(rom.transport_driver, "main", fom_main)
    monkeypatch.setattr(rom, "_ensure_production_context", lambda: None)
    monkeypatch.setattr(rom, "NonlinearManifoldReducedModel", FakeModel)

    with pytest.raises(StopAfterLoad):
        rom.main()

    assert calls["fom"] == expected_fom_calls
    assert len(calls["load"]) == 1
    assert calls["load"][0]["solution_path"] == rom.SOLUTION_PATH
