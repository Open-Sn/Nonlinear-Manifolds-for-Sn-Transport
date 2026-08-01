from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import Transport_Driver_Benchmark_1D as legacy_fom
from one_d.config import (
    InitialConditionConfig,
    OneDConfig,
    OutputConfig,
    ProblemConfig,
    RomConfig,
    TimeIntegrationConfig,
    load_config,
)
from one_d.fom import (
    build_time_array,
    inspect_snapshot,
    save_snapshot,
    validate_fom_solution,
)
from one_d.problem import (
    assemble_operators,
    build_problem,
    construct_boundary_values,
    construct_initial_condition,
)
import one_d.provenance as provenance
from one_d.provenance import create_run_directory, load_manifest
from one_d.workflows import (
    dry_run_fom,
    dry_run_rom,
    execute_fom_workflow,
    execute_rom_workflow,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEGACY_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"
FOM_SCRIPT = REPOSITORY_ROOT / "scripts" / "1d" / "generate_fom.py"
ROM_SCRIPT = REPOSITORY_ROOT / "scripts" / "1d" / "run_rom.py"
INSPECT_SCRIPT = REPOSITORY_ROOT / "scripts" / "1d" / "inspect_snapshot.py"


@pytest.fixture
def tiny_config():
    return OneDConfig(
        schema_version="1.0.0",
        name="tiny_workflow_test",
        description="Fast test-only configuration",
        problem=ProblemConfig(
            region_widths=(1.0, 1.0, 1.0),
            cells_per_region=(2, 2, 2),
            sigma_t=(0.0, 1.0, 0.0),
            sigma_s=(0.0, 0.99, 0.0),
            angular_ordinates=4,
            inflow_boundary="left",
            inflow_direction="most_normal",
            inflow_amplitude=1.0,
            initial_condition=InitialConditionConfig(kind="zero"),
            particle_velocity=1.0,
        ),
        time=TimeIntegrationConfig(
            initial_time=0.0,
            final_time=0.05,
            output_spacing=0.01,
            fom_method="Radau",
            fom_absolute_tolerance=1.0e-10,
            fom_relative_tolerance=1.0e-8,
            rom_method="Radau",
            rom_absolute_tolerance=1.0e-12,
            rom_relative_tolerance=1.0e-9,
            training_end_time=0.03,
        ),
        rom=RomConfig(
            latent_dimension=3,
            lifting_dimension=3,
            embedding_type="tensorial",
            streaming_operators="projected",
            lifting_regularization=1.0e-7,
            linear_inference_regularization=0.0,
            quadratic_inference_regularization_linear=1.0e-5,
            quadratic_inference_regularization_elementwise=1.0e-4,
            quadratic_inference_regularization_tensorial=1.0e-5,
            nonlinear_inference_tolerance=1.0e-6,
            nonlinear_inference_maximum_iterations=100000,
            historical_sequence=(
                "linear:projected",
                "tensorial:projected",
                "elementwise:projected",
                "linear:inferred",
                "tensorial:inferred",
                "elementwise:inferred",
            ),
        ),
        output=OutputConfig(
            snapshot_filename="solutionDG1_A4_T0.05_Nt6_Nx6_tiny.npy",
            output_root="results/1d",
            reuse_existing_snapshot=True,
            allow_overwrite=False,
        ),
    )


def test_configured_tiny_problem_matches_production_assembly_contract(
    tiny_config, tiny_transport_operators
):
    problem = build_problem(tiny_config)
    operators = assemble_operators(problem)
    reference = tiny_transport_operators

    np.testing.assert_allclose(problem.quadrature.mu_q, reference.quadrature.mu_q)
    np.testing.assert_array_equal(problem.mesh.x, reference.mesh.x)
    np.testing.assert_array_equal(problem.mesh.cell2mat, reference.mesh.cell2mat)
    np.testing.assert_allclose(operators.mass.toarray(), reference.mass.toarray())
    np.testing.assert_allclose(
        operators.streaming.toarray(), reference.streaming.toarray()
    )
    np.testing.assert_allclose(
        operators.total_interaction.toarray(), reference.total.toarray()
    )
    np.testing.assert_allclose(
        operators.scattering.toarray(), reference.scattering.toarray()
    )
    np.testing.assert_allclose(operators.system.toarray(), reference.operator.toarray())
    expected_boundary = reference.boundary @ np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(operators.boundary_source, expected_boundary)


def test_root_helpers_match_configured_stages_on_small_data():
    config = load_config(LEGACY_CONFIG_PATH)
    problem = build_problem(config)
    coordinates = np.array([0.0, 0.05, 0.1, 0.2, 1.0])
    configured_problem = replace(problem, dof_coordinates=coordinates)

    np.testing.assert_array_equal(problem.quadrature.mu_q, legacy_fom.myAQ.mu_q)
    np.testing.assert_array_equal(problem.mesh.x, legacy_fom.myMESH.x)
    np.testing.assert_array_equal(problem.dof_coordinates, legacy_fom.xx)
    np.testing.assert_array_equal(
        construct_initial_condition(configured_problem),
        legacy_fom.make_production_initial_condition(coordinates),
    )
    np.testing.assert_array_equal(
        construct_boundary_values(problem),
        legacy_fom.make_psi_bc_dir(
            lambda current_time: 1.0, left="most_grazing", right=None
        )(0.0),
    )
    np.testing.assert_array_equal(build_time_array(config), legacy_fom.PRODUCTION_TIME_STEPS)


def test_root_and_configured_solution_validators_accept_same_tiny_result(tiny_config):
    time = build_time_array(tiny_config)
    result = SimpleNamespace(
        success=True,
        message="ok",
        t=time.copy(),
        y=np.zeros(tiny_config.expected_snapshot_shape),
    )
    assert validate_fom_solution(result, time, 48) is result
    assert legacy_fom.validate_solve_ivp_result(
        result, time, 48, "tiny compatibility", expected_final_time=0.05
    ) is result


def test_fom_dry_run_is_matrix_free_and_writes_nothing(tiny_config, tmp_path):
    snapshot = tmp_path / tiny_config.output.snapshot_filename
    report = dry_run_fom(tiny_config, snapshot_path=snapshot)

    assert report.action == "would_solve"
    assert report.expected_snapshot_shape == (48, 6)
    assert report.estimated_raw_snapshot_bytes == 48 * 6 * 8
    assert report.snapshot_exists is False
    assert report.assembles_operators is False
    assert report.solves is False
    assert report.writes_files is False
    assert list(tmp_path.iterdir()) == []


def test_fom_dry_run_reuses_valid_snapshot_and_refuses_overwrite(
    tiny_config, tmp_path
):
    snapshot = tmp_path / tiny_config.output.snapshot_filename
    np.save(snapshot, np.zeros(tiny_config.expected_snapshot_shape))

    assert dry_run_fom(tiny_config, snapshot_path=snapshot).action == "reuse"
    refusing_config = replace(
        tiny_config,
        output=replace(
            tiny_config.output,
            reuse_existing_snapshot=False,
            allow_overwrite=False,
        ),
    )
    assert (
        dry_run_fom(refusing_config, snapshot_path=snapshot).action
        == "refuse_existing_snapshot"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_snapshot(
            snapshot,
            np.zeros(tiny_config.expected_snapshot_shape),
            tiny_config,
        )


def test_rom_dry_run_distinguishes_valid_and_missing_snapshots(tiny_config, tmp_path):
    snapshot = tmp_path / tiny_config.output.snapshot_filename
    missing = dry_run_rom(
        tiny_config,
        model="tensorial",
        operators="inferred",
        input_snapshot=snapshot,
    )
    assert missing.action == "refuse_missing_snapshot"
    np.save(snapshot, np.zeros(tiny_config.expected_snapshot_shape))
    present = dry_run_rom(
        tiny_config,
        model="element-wise",
        operators="projected",
        input_snapshot=snapshot,
    )
    assert present.action == "would_run_selected_model"
    assert present.model == "elementwise"
    assert present.writes_files is False


def test_workflow_execution_requires_explicit_authorization(tiny_config, tmp_path):
    before = list(tmp_path.iterdir())
    with pytest.raises(PermissionError, match="execute=True"):
        execute_fom_workflow(tiny_config, run_directory=tmp_path / "fom")
    with pytest.raises(PermissionError, match="execute=True"):
        execute_rom_workflow(
            tiny_config,
            model="linear",
            operators="projected",
            input_snapshot=tmp_path / "missing.npy",
            run_directory=tmp_path / "rom",
        )
    assert list(tmp_path.iterdir()) == before


def test_snapshot_inspection_reports_compatibility_and_hash(tiny_config, tmp_path):
    snapshot = tmp_path / tiny_config.output.snapshot_filename
    np.save(snapshot, np.zeros(tiny_config.expected_snapshot_shape))
    inspection = inspect_snapshot(snapshot, tiny_config, include_sha256=True)

    assert inspection.path == str(snapshot)
    assert inspection.exists is True
    assert inspection.shape == (48, 6)
    assert inspection.dtype == "float64"
    assert inspection.finite is True
    assert inspection.compatible is True
    assert inspection.expected_shape == (48, 6)
    assert inspection.sha256 is not None and len(inspection.sha256) == 64
    assert inspection.time_count == 6
    assert inspection.initial_time == 0.0
    assert inspection.final_time == 0.05
    assert inspection.output_spacing == 0.01


@pytest.mark.parametrize("dirty", [False, True])
def test_run_directory_manifest_records_git_state(
    monkeypatch, tiny_config, tmp_path, dirty
):
    monkeypatch.setattr(
        provenance,
        "_git_metadata",
        lambda repository_root=None: {"commit": "abc123", "dirty": dirty},
    )
    run = create_run_directory(
        tiny_config,
        run_directory=tmp_path / f"run-{dirty}",
        config_source="tiny.json",
        execution_stage="test",
    )
    manifest = load_manifest(run)

    assert run.config_path.is_file()
    assert run.manifest_path.is_file()
    assert all(path.is_dir() for path in (run.logs, run.data, run.metrics, run.figures))
    assert manifest["git"] == {"commit": "abc123", "dirty": dirty}
    assert manifest["configuration_checksum_sha256"] == tiny_config.checksum()
    assert manifest["configuration_canonical_json"] == tiny_config.canonical_json()
    assert manifest["snapshot"]["filename"] == tiny_config.output.snapshot_filename
    assert manifest["execution"]["stage"] == "test"


def _subprocess_environment():
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


@pytest.mark.parametrize(
    ("script", "arguments", "expected_action"),
    [
        (
            FOM_SCRIPT,
            ["--config", str(LEGACY_CONFIG_PATH), "--dry-run"],
            "would_solve",
        ),
        (
            ROM_SCRIPT,
            [
                "--config",
                str(LEGACY_CONFIG_PATH),
                "--model",
                "tensorial",
                "--operators",
                "inferred",
                "--dry-run",
            ],
            "refuse_missing_snapshot",
        ),
    ],
)
def test_production_cli_dry_runs_write_nothing(
    script, arguments, expected_action, tmp_path
):
    result = subprocess.run(
        [sys.executable, str(script), *arguments],
        cwd=tmp_path,
        env=_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["action"] == expected_action
    assert report["assembles_operators"] is False
    assert report["solves"] is False
    assert report["writes_files"] is False
    assert list(tmp_path.iterdir()) == []


def test_snapshot_inspection_cli_is_read_only(tiny_config, tmp_path):
    config_path = tmp_path / "tiny.json"
    snapshot_path = tmp_path / tiny_config.output.snapshot_filename
    config_path.write_text(json.dumps(tiny_config.to_dict()), encoding="utf-8")
    np.save(snapshot_path, np.zeros(tiny_config.expected_snapshot_shape))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            str(INSPECT_SCRIPT),
            str(snapshot_path),
            "--config",
            str(config_path),
            "--sha256",
        ],
        cwd=tmp_path,
        env=_subprocess_environment(),
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["compatible"] is True
    assert report["shape"] == [48, 6]
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_new_workflow_code_has_no_network_dependencies():
    paths = [
        *sorted((REPOSITORY_ROOT / "one_d").glob("*.py")),
        *sorted((REPOSITORY_ROOT / "scripts" / "1d").glob("*.py")),
    ]
    forbidden = ("import requests", "import socket", "import urllib", "import http")
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path
