import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from one_d.config import load_config
from one_d.publication_artifacts import (
    build_figure_data_bundle,
    create_publication_run_directory,
    update_publication_run,
    write_npz_artifact,
)
from one_d.publication_experiments import (
    AUTHOR_CONFIRMED_SIGMOID_PROVENANCE,
    GOLDEN_CONTENT_CHECKSUM,
    load_publication_catalog,
    resolve_case_configuration,
)
from one_d.publication_plotting import (
    manuscript_plot_plan,
    plot_manuscript_figure,
    production_ordinates,
    production_spatial_dof_coordinates,
    split_direction_major_state,
    validate_manuscript_figure_bundle,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PLOTTING_SCRIPT = REPOSITORY_ROOT / "scripts" / "1d" / "plot_publication_figures.py"
PLOTTING_MODULE = REPOSITORY_ROOT / "one_d" / "publication_plotting.py"
PROVENANCE_PATH = (
    REPOSITORY_ROOT
    / "configs"
    / "1d"
    / "publication"
    / "initial_condition_provenance.json"
)
GOLDEN_MANIFEST_PATH = REPOSITORY_ROOT / "tests" / "golden" / "tiny_1d_manifest.json"


@pytest.fixture(scope="module")
def catalog():
    return load_publication_catalog()


def _complete_case(run, *, diagnostics=None):
    update_publication_run(
        run,
        metrics={"synthetic_plotting_fixture": True},
        diagnostics=diagnostics or {"synthetic_plotting_fixture": True},
        solver={"status": "synthetic_complete", "success": True, "message": None},
    )


def _make_figure1_bundle(tmp_path, catalog):
    case = catalog.get("fig1_pod_reducibility")
    config = resolve_case_configuration(case)
    run = create_publication_run_directory(
        catalog,
        case,
        config,
        input_snapshot="synthetic/" + case.required_input_snapshot,
        input_snapshot_checksum="a" * 64,
        output_root=tmp_path / "artifacts",
        run_id="synthetic-figure1",
    )
    dimensions = np.arange(1, 701)
    unresolved = np.maximum(np.logspace(-1.0, -17.0, dimensions.size), 5.0e-16)
    write_npz_artifact(
        run,
        "pod_spectrum.npz",
        {
            "basis_dimensions": dimensions,
            "highlighted_latent_dimension": np.array([16]),
            "highlighted_lifting_dimension": np.array([548]),
            "highlighted_total_dimension": np.array([564]),
            "pod_eigenvalues": unresolved.copy(),
            "retained_energy_fraction": 1.0 - unresolved,
            "unresolved_energy_fraction": unresolved,
        },
    )
    _complete_case(run)
    return build_figure_data_bundle(
        "Figure 1",
        [run.root],
        catalog=catalog,
        output_directory=tmp_path / "figure1-bundle",
    ).root


def _make_figure2_case(tmp_path, catalog, case_id, scale):
    case = catalog.get(case_id)
    config = resolve_case_configuration(case)
    run = create_publication_run_directory(
        catalog,
        case,
        config,
        input_snapshot="synthetic/" + case.required_input_snapshot,
        input_snapshot_checksum="b" * 64,
        output_root=tmp_path / "artifacts",
        run_id="synthetic-figure2",
    )
    coordinates = production_spatial_dof_coordinates()
    blocks = [
        (index + 1.0) * (0.2 + coordinates) for index in range(4)
    ]
    reference = np.concatenate(blocks)
    reconstruction = reference - scale * np.sin(np.arange(reference.size) / 31.0)
    write_npz_artifact(
        run,
        "fields.npz",
        {
            "field_time": np.array([2.5]),
            "field_time_index": np.array([2500]),
            "dof_index": np.arange(reference.size),
            "fom_angular_flux": reference,
            "rom_angular_flux": reconstruction,
            "discrepancy": reference - reconstruction,
        },
    )
    time = np.linspace(0.0, 10.0, config.time.output_count)
    write_npz_artifact(
        run,
        "error_history.npz",
        {
            "time": time,
            "instantaneous_normalized_mass_error": 1.0e-3 + scale * np.exp(-time),
            "training_end_time": np.array([7.5]),
        },
    )
    _complete_case(run)
    return run


def _make_figure2_bundle(tmp_path, catalog, *, complete=True):
    cases = [
        ("fig2_linear_projected", 3.0e-2),
        ("fig2_elementwise_projected", 2.0e-2),
        ("fig2_tensorial_projected", 1.0e-2),
    ]
    if not complete:
        cases = cases[:1]
    roots = [
        _make_figure2_case(tmp_path, catalog, case_id, scale).root
        for case_id, scale in cases
    ]
    return build_figure_data_bundle(
        "Figure 2",
        roots,
        catalog=catalog,
        output_directory=tmp_path / "figure2-bundle",
    ).root


def _run_script(arguments):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["MPLCONFIGDIR"] = str(REPOSITORY_ROOT / ".pytest_cache" / "mpl")
    return subprocess.run(
        [sys.executable, str(PLOTTING_SCRIPT), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _tree_checksum(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def test_author_confirmed_sigmoid_policy_and_no_zero_publication_configuration():
    recorded = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    for key, value in AUTHOR_CONFIRMED_SIGMOID_PROVENANCE.items():
        assert recorded[key] == value
    assert recorded["zero_initial_condition_production_configuration_maintained"] is False
    config = load_config(REPOSITORY_ROOT / recorded["authoritative_configuration"])
    assert config.problem.initial_condition.kind == "localized_sigmoid"
    assert config.problem.initial_condition.angular_block == "final"


def test_phase_space_mapping_preserves_dg_and_direction_major_order():
    coordinates = production_spatial_dof_coordinates()
    ordinates = production_ordinates()
    assert coordinates.shape == (1500,)
    np.testing.assert_array_equal(coordinates[:4], [0.0, 0.004, 0.004, 0.008])
    assert coordinates[0] == 0.0 and coordinates[-1] == 3.0
    assert ordinates.tolist() == sorted(ordinates.tolist())

    blocks = np.vstack(
        [np.arange(1500, dtype=float) + 10_000.0 * index for index in range(4)]
    )
    mapped = split_direction_major_state(blocks.reshape(-1))
    np.testing.assert_array_equal(mapped.spatial_coordinates, coordinates)
    np.testing.assert_array_equal(mapped.ordinates, ordinates)
    np.testing.assert_array_equal(mapped.direction_blocks, blocks)


def test_figure1_generation_metadata_dry_run_and_overwrite_refusal(tmp_path, catalog):
    bundle = _make_figure1_bundle(tmp_path, catalog)
    validated = validate_manuscript_figure_bundle(
        bundle, expected_figure="Figure 1"
    )
    output = tmp_path / "manuscript-figure1"
    plan = manuscript_plot_plan(validated, output_directory=output)
    assert plan["writes_files"] is False
    assert plan["outputs"][0].endswith("figure1_manuscript.png")
    assert not output.exists()

    paths = plot_manuscript_figure(
        bundle,
        output_directory=output,
        expected_figure="Figure 1",
    )
    assert {path.suffix for path in paths[:2]} == {".png", ".pdf"}
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    details = metadata["plot_details"]
    assert details["displayed_x_limits"] == [0.0, 700.0]
    assert details["highlighted_dimensions"] == {
        "N_q": 548,
        "N_r": 16,
        "N_r_plus_N_q": 564,
    }
    assert metadata["source_bundle"]["checksum_sha256"]
    assert metadata["initial_condition_provenance"] == (
        AUTHOR_CONFIRMED_SIGMOID_PROVENANCE
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        plot_manuscript_figure(bundle, output_directory=output)


def test_figure2_complete_series_markers_discrepancy_and_diagnostic_preservation(
    tmp_path, catalog
):
    bundle = _make_figure2_bundle(tmp_path, catalog)
    diagnostic = tmp_path / "existing-diagnostic"
    diagnostic.mkdir()
    (diagnostic / "sentinel.png").write_bytes(b"unchanged diagnostic")
    before = _tree_checksum(diagnostic)

    output = tmp_path / "manuscript-figure2"
    paths = plot_manuscript_figure(
        bundle,
        output_directory=output,
        expected_figure="Figure 2",
    )
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
    metadata = json.loads(paths[2].read_text(encoding="utf-8"))
    details = metadata["plot_details"]
    assert set(metadata["case_ids"]) == {
        "fig2_linear_projected",
        "fig2_elementwise_projected",
        "fig2_tensorial_projected",
    }
    assert details["field_time"] == 2.5
    assert details["training_extrapolation_boundary"] == 7.5
    assert details["field_time_marker"] == {
        "color": "black",
        "style": "dashed",
        "time": 2.5,
    }
    assert details["training_boundary_marker"] == {
        "color": "red",
        "style": "dashed",
        "time": 7.5,
    }
    assert details["discrepancy_definition"] == "reference_minus_reconstruction"
    assert details["physical_x_limits"] == [0.0, 3.0]
    assert len(details["phase_space_mapping"]["ordinates_in_stored_order"]) == 4
    assert _tree_checksum(diagnostic) == before


def test_dry_run_command_writes_nothing(tmp_path, catalog):
    bundle = _make_figure1_bundle(tmp_path, catalog)
    output = tmp_path / "dry-run-output"
    completed = _run_script(
        [
            "--figure",
            "1",
            "--source-bundle",
            str(bundle),
            "--layout",
            "manuscript",
            "--output-dir",
            str(output),
            "--dry-run",
        ]
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["writes_files"] is False
    assert not output.exists()


def test_incomplete_figure2_bundle_is_refused(tmp_path, catalog):
    bundle = _make_figure2_bundle(tmp_path, catalog, complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        validate_manuscript_figure_bundle(bundle, expected_figure="Figure 2")


def test_plotting_sources_do_not_import_or_invoke_scientific_entry_points():
    forbidden_imports = {
        "one_d.fom",
        "one_d.rom",
        "Transport_Driver_Benchmark_1D",
        "Nonlinear_Manifold_ROM",
    }
    for path in (PLOTTING_SCRIPT, PLOTTING_MODULE):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not forbidden_imports.intersection(imported)
        for forbidden_call in (
            "solve_fom",
            "run_selected_rom",
            "execute_publication_case",
            "compute_pod_data",
        ):
            assert forbidden_call not in source


def test_independent_golden_checksum_remains_unchanged():
    manifest = json.loads(GOLDEN_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert manifest["content_checksum"]["sha256"] == GOLDEN_CONTENT_CHECKSUM
