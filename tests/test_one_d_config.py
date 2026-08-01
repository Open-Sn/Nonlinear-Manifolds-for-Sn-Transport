from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import Nonlinear_Manifold_ROM as legacy_rom
import Transport_Driver_Benchmark_1D as legacy_fom
from one_d.config import OneDConfig, load_config
from one_d.fom import build_time_array


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"


@pytest.fixture(scope="module")
def legacy_config():
    return load_config(LEGACY_CONFIG_PATH)


def test_legacy_json_preserves_protected_production_defaults(legacy_config):
    problem = legacy_config.problem
    time = legacy_config.time
    rom = legacy_config.rom

    assert problem.region_widths == (1.0, 1.0, 1.0)
    assert problem.cells_per_region == (250, 250, 250)
    assert problem.sigma_t == (0.0, 1.0, 0.0)
    assert problem.sigma_s == (0.0, 0.99, 0.0)
    assert problem.angular_ordinates == 4
    assert problem.inflow_boundary == "left"
    assert problem.inflow_direction == "most_normal"
    assert problem.inflow_amplitude == 1.0
    assert problem.particle_velocity == 1.0
    assert problem.initial_condition.kind == "localized_sigmoid"
    assert problem.initial_condition.transition_location == 0.1
    assert problem.initial_condition.steepness == 100.0
    assert problem.initial_condition.amplitude == 1.0
    assert problem.initial_condition.angular_block == "final"

    assert time.initial_time == 0.0
    assert time.final_time == legacy_fom.TT == 10.0
    assert time.output_spacing == legacy_fom.dt == 0.001
    assert time.output_count == legacy_fom.NN == 10001
    assert (time.fom_method, time.fom_absolute_tolerance, time.fom_relative_tolerance) == (
        "Radau",
        1.0e-9,
        1.0e-12,
    )
    assert (time.rom_method, time.rom_absolute_tolerance, time.rom_relative_tolerance) == (
        "Radau",
        1.0e-12,
        1.0e-9,
    )
    assert time.training_end_time == 7.5

    assert rom.latent_dimension == 16
    assert rom.lifting_dimension == 364
    assert rom.lifting_regularization == 0.0001875
    assert rom.linear_inference_regularization == 0.0
    assert rom.quadratic_inference_regularization_linear == 1.0e-5
    assert rom.quadratic_inference_regularization_elementwise == 12.0
    assert rom.quadratic_inference_regularization_tensorial == 0.012
    assert rom.nonlinear_inference_tolerance == 1.0e-6
    assert rom.nonlinear_inference_maximum_iterations == (
        legacy_rom.DEFAULT_MAX_INFERENCE_ITERATIONS
    )
    assert legacy_config.output.snapshot_filename == legacy_fom.SOLUTION_PATH
    assert legacy_config.expected_snapshot_shape == (6000, 10001)
    assert legacy_config.output.reuse_existing_snapshot is True
    assert legacy_config.output.allow_overwrite is False


def test_canonical_configuration_serialization_and_checksum_are_deterministic(
    legacy_config,
):
    canonical = legacy_config.canonical_json()
    reconstructed = OneDConfig.from_dict(json.loads(canonical))

    assert reconstructed.canonical_json() == canonical
    assert reconstructed.checksum() == legacy_config.checksum()
    assert (
        legacy_config.checksum()
        == "cc442174134332f4b722cfa65ef179e1abc350c3e27e342a8bfeb184aa1b2759"
    )
    assert "NaN" not in canonical


def test_configured_time_grid_matches_historical_production_grid(legacy_config):
    time = build_time_array(legacy_config)
    np.testing.assert_array_equal(time, legacy_fom.PRODUCTION_TIME_STEPS)
    assert time[0] == 0.0
    assert time[-1] == 10.0
    assert time.size == 10001


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: replace(
                config,
                problem=replace(config.problem, region_widths=(-1.0, 1.0, 1.0)),
            ),
            "widths",
        ),
        (
            lambda config: replace(
                config,
                problem=replace(config.problem, cells_per_region=(250, 0, 250)),
            ),
            "cell counts",
        ),
        (
            lambda config: replace(
                config,
                problem=replace(config.problem, sigma_t=(0.0, 1.0)),
            ),
            "matching lengths",
        ),
        (
            lambda config: replace(
                config,
                time=replace(config.time, final_time=0.0),
            ),
            "final_time",
        ),
        (
            lambda config: replace(
                config,
                time=replace(config.time, output_spacing=-0.001),
            ),
            "output_spacing",
        ),
        (
            lambda config: replace(
                config,
                rom=replace(config.rom, embedding_type="invalid"),
            ),
            "embedding_type",
        ),
        (
            lambda config: replace(
                config,
                rom=replace(config.rom, latent_dimension=0),
            ),
            "latent_dimension",
        ),
        (
            lambda config: replace(
                config,
                problem=replace(config.problem, particle_velocity=np.nan),
            ),
            "finite",
        ),
        (
            lambda config: replace(
                config,
                output=replace(
                    config.output,
                    snapshot_filename=(
                        "solutionDG1_A4_T10_Nt10000_Nx750_continuous_bis.npy"
                    ),
                ),
            ),
            "time-count",
        ),
    ],
)
def test_configuration_validation_rejects_invalid_values(
    legacy_config, mutation, message
):
    with pytest.raises(ValueError, match=message):
        mutation(legacy_config)
