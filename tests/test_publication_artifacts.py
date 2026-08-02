import ast
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import one_d.publication_artifacts as publication_artifacts
import one_d.rom as rom
from one_d.publication_artifacts import (
    build_figure_data_bundle,
    create_publication_run_directory,
    plot_figure_data_bundle,
    update_publication_run,
    validate_publication_artifact,
    write_npz_artifact,
)
from one_d.publication_experiments import (
    dry_run_publication_case,
    load_publication_catalog,
    resolve_base_configuration,
    resolve_case_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY_ROOT / "scripts" / "1d"


@pytest.fixture(scope="module")
def catalog():
    return load_publication_catalog()


def _make_synthetic_figure2_artifact(tmp_path, catalog):
    case = catalog.get("fig2_linear_projected")
    config = resolve_case_configuration(case)
    run = create_publication_run_directory(
        catalog,
        case,
        config,
        input_snapshot="synthetic/" + case.required_input_snapshot,
        input_snapshot_checksum="a" * 64,
        output_root=tmp_path,
        run_id="synthetic-run",
    )
    dofs = config.problem.phase_space_dofs
    count = config.time.output_count
    fom = np.linspace(0.0, 1.0, dofs)
    rom = 0.99 * fom
    write_npz_artifact(
        run,
        "fields.npz",
        {
            "field_time": np.array([2.5]),
            "dof_index": np.arange(dofs),
            "fom_angular_flux": fom,
            "rom_angular_flux": rom,
            "discrepancy": fom - rom,
        },
    )
    write_npz_artifact(
        run,
        "error_history.npz",
        {
            "time": np.linspace(0.0, 10.0, count),
            "instantaneous_normalized_mass_error": np.linspace(1.0e-2, 1.0e-4, count),
            "training_end_time": np.array([7.5]),
        },
    )
    update_publication_run(
        run,
        metrics={"instantaneous_error_history_metric_id": "instantaneous_steady_state_normalized_mass_error"},
        diagnostics={"synthetic_fixture": True},
        solver={"status": "synthetic_complete", "success": True, "message": None},
    )
    return run


def _make_synthetic_figure4_artifact(tmp_path, catalog, case):
    config = resolve_base_configuration(case)
    run = create_publication_run_directory(
        catalog,
        case,
        config,
        input_snapshot="synthetic/" + case.required_input_snapshot,
        input_snapshot_checksum="b" * 64,
        output_root=tmp_path,
        run_id="synthetic-run",
    )
    write_npz_artifact(
        run,
        "convergence_data.npz",
        {
            "model_dimension": np.array([case.latent_dimension]),
            "relative_convergence_metric": np.array([1.0e-3]),
            "online_runtime_seconds": np.array([0.01]),
            "online_speedup": np.array([10.0]),
            "ode_function_evaluations": np.array([20]),
        },
    )
    update_publication_run(
        run,
        metrics={"synthetic_author_defined_metric": True},
        diagnostics={"synthetic_fixture": True},
        solver={"status": "synthetic_complete", "success": True, "message": None},
    )
    return run


def _run_script(arguments, *, cwd=REPOSITORY_ROOT):
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=cwd,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_result_artifact_schema_validates_json_npz_and_provenance(tmp_path, catalog):
    run = _make_synthetic_figure2_artifact(tmp_path, catalog)
    manifest = validate_publication_artifact(run.root, catalog=catalog)
    assert manifest["benchmark_variant"] == "legacy_sigmoid"
    assert manifest["manuscript_deviation"]["initial_condition"]["status"] == "intentional_repository_deviation"
    assert manifest["input_snapshot"]["checksum_sha256"] == "a" * 64
    assert manifest["model"]["type"] == "linear"
    assert manifest["model"]["operator_construction"] == "projected"
    assert manifest["model"]["regularization"] == {
        "gamma": None,
        "lambda_L": None,
        "lambda_Q": None,
    }
    assert len(manifest["arrays"]) == 8


def test_publication_rom_execution_scales_ridges_by_training_count(
    monkeypatch, catalog
):
    case = catalog.get("fig2_elementwise_projected")
    config = resolve_case_configuration(case)
    captured = {}

    class FieldHistory:
        def __getitem__(self, key):
            assert key == (slice(None), 2500)
            return np.zeros(config.problem.phase_space_dofs)

    def fake_run_selected_rom(config_argument, snapshot_path, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            reconstructed_state=FieldHistory(),
            time=np.array([0.0]),
            errors=np.array([0.0]),
            diagnostics={"solver_success": True, "solver_message": "ok"},
        )

    monkeypatch.setattr(rom, "run_selected_rom", fake_run_selected_rom)
    monkeypatch.setattr(
        publication_artifacts.np,
        "load",
        lambda *args, **kwargs: FieldHistory(),
    )
    monkeypatch.setattr(
        publication_artifacts,
        "write_npz_artifact",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        publication_artifacts,
        "update_publication_run",
        lambda *args, **kwargs: None,
    )

    publication_artifacts._execute_rom_case(
        SimpleNamespace(), case, config, Path("validated-snapshot.npy")
    )

    assert captured["regularization_scale"] == 7501.0


def test_result_artifact_rejects_benchmark_or_shape_tampering(tmp_path, catalog):
    run = _make_synthetic_figure2_artifact(tmp_path, catalog)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_variant"] = "paper_zero"
    run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="benchmark_variant"):
        validate_publication_artifact(run.root, catalog=catalog)

    run = _make_synthetic_figure2_artifact(tmp_path / "second", catalog)
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["arrays"] if item["array_name"] == "discrepancy")
    record["shape"] = [1]
    run.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata mismatch"):
        validate_publication_artifact(run.root, catalog=catalog)


def test_figure_data_and_plotting_use_only_synthetic_validated_artifacts(tmp_path, catalog):
    pytest.importorskip("matplotlib")
    run = _make_synthetic_figure2_artifact(tmp_path / "runs", catalog)
    bundle = build_figure_data_bundle(
        "Figure 2",
        [run.root],
        catalog=catalog,
        output_directory=tmp_path / "bundle",
    )
    assert bundle.case_ids == ("fig2_linear_projected",)
    assert not bundle.complete
    metadata = json.loads(bundle.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "partial_input_set"
    assert metadata["benchmark_variant"] == "legacy_sigmoid"
    assert metadata["case_set_complete"] is False
    assert metadata["series_membership"] == ["linear"]
    assert metadata["missing_series"] == ["elementwise", "tensorial"]
    assert metadata["complete_publication_reproduction"] is False
    paths = plot_figure_data_bundle(bundle.root, output_directory=tmp_path / "plots")
    assert paths
    assert all(path.is_file() for path in paths)
    plot_metadata = json.loads((tmp_path / "plots" / "plot_metadata.json").read_text())
    assert plot_metadata["benchmark_variant"] == "legacy_sigmoid"
    assert plot_metadata["complete_publication_reproduction"] is False


def test_figure4_bundle_requires_all_six_series(tmp_path, catalog, monkeypatch):
    monkeypatch.setattr(
        "one_d.publication_artifacts._git_metadata",
        lambda _repository_root=None: {"commit": "synthetic", "dirty": False},
    )
    cases = [case for case in catalog.cases if case.figure == "Figure 4"]
    runs = [
        _make_synthetic_figure4_artifact(tmp_path / "runs", catalog, case)
        for case in cases
    ]
    complete = build_figure_data_bundle(
        "Figure 4",
        [run.root for run in runs],
        catalog=catalog,
        output_directory=tmp_path / "complete-bundle",
    )
    assert complete.complete
    metadata = json.loads(complete.metadata_path.read_text(encoding="utf-8"))
    assert metadata["series_membership"] == [
        "inferred_elementwise",
        "inferred_linear",
        "inferred_tensorial",
        "projected_elementwise",
        "projected_linear",
        "projected_tensorial",
    ]
    assert metadata["missing_series"] == []

    for operators in ("projected", "inferred"):
        partial_roots = [
            run.root
            for run, case in zip(runs, cases)
            if not (case.model_type == "linear" and case.operator_construction == operators)
        ]
        partial = build_figure_data_bundle(
            "Figure 4",
            partial_roots,
            catalog=catalog,
            output_directory=tmp_path / f"missing-{operators}-linear",
        )
        assert not partial.complete
        partial_metadata = json.loads(partial.metadata_path.read_text(encoding="utf-8"))
        assert f"{operators}_linear" in partial_metadata["missing_series"]
        assert partial_metadata["complete_publication_reproduction"] is False


def test_listing_inspection_and_dry_runs_are_read_only(tmp_path):
    output_root = tmp_path / "results"
    listing = _run_script([str(SCRIPTS / "list_publication_cases.py"), "--json"])
    assert listing.returncode == 0, listing.stderr
    listing_data = json.loads(listing.stdout)
    assert listing_data["case_count"] == 57
    assert listing_data["writes_files"] is False
    for case_id in (
        "fig2_tensorial_projected",
        "fig3_tensorial_inferred",
        "fig4_linear_projected_nr32",
        "fig4_linear_inferred_nr32",
        "fig4_tensorial_inferred_nr32",
    ):
        inspection = _run_script([str(SCRIPTS / "inspect_publication_case.py"), case_id])
        assert inspection.returncode == 0, inspection.stderr
        data = json.loads(inspection.stdout)
        assert data["dry_run"]["benchmark_variant"] == "legacy_sigmoid"
        assert data["dry_run"]["assembles_operators"] is False
        assert data["dry_run"]["writes_files"] is False
        if case_id == "fig4_linear_projected_nr32":
            assert data["case"]["lifting_dimension"] is None
            assert data["case"]["lifting_regularization_gamma"] is None
            assert data["case"]["lambda_L"] is None
            assert data["case"]["lambda_Q"] is None
        if case_id == "fig4_linear_inferred_nr32":
            assert data["case"]["lifting_dimension"] is None
            assert data["case"]["lifting_regularization_gamma"] is None
            assert data["case"]["lambda_L"] == 0.0
            assert data["case"]["lambda_Q"] is None
    dry = _run_script(
        [
            str(SCRIPTS / "run_publication_case.py"),
            "fig3_tensorial_inferred",
            "--dry-run",
            "--output-root",
            str(output_root),
        ]
    )
    assert dry.returncode == 0, dry.stderr
    data = json.loads(dry.stdout)
    assert data["action"] == "refuse_missing_snapshot"
    assert data["benchmark_variant"] == "legacy_sigmoid"
    assert data["writes_files"] is False
    default_safe = _run_script(
        [
            str(SCRIPTS / "run_publication_case.py"),
            "fig3_tensorial_inferred",
            "--output-root",
            str(output_root),
        ]
    )
    assert default_safe.returncode == 0, default_safe.stderr
    assert json.loads(default_safe.stdout)["writes_files"] is False
    under_specified = _run_script(
        [
            str(SCRIPTS / "run_publication_case.py"),
            "fig4_tensorial_inferred_nr32",
            "--dry-run",
            "--output-root",
            str(output_root),
        ]
    )
    assert under_specified.returncode == 0, under_specified.stderr
    data = json.loads(under_specified.stdout)
    assert data["action"] == "refuse_under_specified"
    assert any("author-selected gamma" in item for item in data["missing_information"])
    for case_id, expected_lambda_L in (
        ("fig4_linear_projected_nr32", None),
        ("fig4_linear_inferred_nr32", 0.0),
    ):
        linear_dry = _run_script(
            [
                str(SCRIPTS / "run_publication_case.py"),
                case_id,
                "--dry-run",
                "--output-root",
                str(output_root),
            ]
        )
        assert linear_dry.returncode == 0, linear_dry.stderr
        linear_data = json.loads(linear_dry.stdout)
        assert linear_data["action"] == "refuse_missing_snapshot"
        assert linear_data["lifting_dimension"] is None
        assert linear_data["lifting_regularization_gamma"] is None
        assert linear_data["lambda_L"] == expected_lambda_L
        assert linear_data["lambda_Q"] is None
        assert linear_data["inference_tolerance"] is None
        assert linear_data["maximum_iterations"] is None
        assert linear_data["missing_information"] == []
        assert linear_data["assembles_operators"] is False
        assert linear_data["solves"] is False
        assert linear_data["writes_files"] is False
    assert not output_root.exists()


def test_execution_refuses_without_flag_snapshot_or_full_specification(tmp_path, catalog):
    case = catalog.get("fig3_tensorial_inferred")
    report = dry_run_publication_case(catalog, case, snapshot_path=tmp_path / "missing.npy")
    assert report.action == "refuse_missing_snapshot"
    output_root = tmp_path / "results"
    no_snapshot = _run_script(
        [
            str(SCRIPTS / "run_publication_case.py"),
            case.case_id,
            "--execute",
            "--snapshot",
            str(tmp_path / "missing.npy"),
            "--output-root",
            str(output_root),
        ]
    )
    assert no_snapshot.returncode == 2
    under_specified = _run_script(
        [
            str(SCRIPTS / "run_publication_case.py"),
            "fig4_tensorial_inferred_nr32",
            "--execute",
            "--output-root",
            str(output_root),
        ]
    )
    assert under_specified.returncode == 2
    incompatible = tmp_path / "incompatible.npy"
    np.save(incompatible, np.zeros((2, 2)))
    wrong_snapshot = _run_script(
        [
            str(SCRIPTS / "run_publication_case.py"),
            case.case_id,
            "--execute",
            "--snapshot",
            str(incompatible),
            "--output-root",
            str(output_root),
        ]
    )
    assert wrong_snapshot.returncode == 2
    assert not output_root.exists()


def test_plotting_script_does_not_import_or_call_scientific_entry_points():
    source = (SCRIPTS / "plot_publication_figures.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not {"one_d.fom", "one_d.rom", "Transport_Driver_Benchmark_1D", "Nonlinear_Manifold_ROM"}.intersection(imported)
    assert "solve_fom" not in source
    assert "run_selected_rom" not in source
    assert "execute_publication_case" not in source


def test_new_publication_layers_have_no_network_execution_dependencies():
    paths = [
        REPOSITORY_ROOT / "one_d" / "publication_experiments.py",
        REPOSITORY_ROOT / "one_d" / "publication_artifacts.py",
        REPOSITORY_ROOT / "one_d" / "publication_metrics.py",
    ]
    forbidden = {"requests", "urllib", "http.client", "socket", "aiohttp"}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        assert not forbidden.intersection(imported)
