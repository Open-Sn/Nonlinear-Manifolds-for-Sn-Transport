import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import Nonlinear_Manifold_ROM as legacy_rom
from Nonlinear_Manifold_ROM import NonlinearManifoldReducedModel
from one_d.figure5 import (
    EXPECTED_DATASET_CHECKSUM,
    NQ_VALUES,
    TRAINING_COUNT,
    build_figure5_bundle,
    figure5_cases,
    figure5_execution_plan,
    validate_figure5_bundle,
)
from one_d.figure5_plotting import plot_figure5_bundle
from one_d.publication_metrics import (
    relative_space_time_l2_error,
    relative_space_time_l2_error_from_energies,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FOM_SECONDS = 662.6050200000172
GOLDEN_CHECKSUM = "91c84e813e5cbfabd0bf0c5be436afc19e64152b7f06c9f1a572a76038108238"


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _case_checksum(case):
    encoded = json.dumps(
        case.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _synthetic_case_result(output_root, run_id, case):
    root = output_root / "figure5_cases" / run_id / case.case_id
    root.mkdir(parents=True)
    if case.model == "best_projection":
        error = 2.0 / case.projection_dimension
        online = None
        speedup = None
    elif case.model == "linear" and case.N_r == 32:
        error = 0.5 if case.operators == "projected" else 0.6
        online = 0.25 if case.operators == "projected" else 0.3
        speedup = FOM_SECONDS / online
    else:
        model_factor = {"linear": 1.3, "elementwise": 1.1, "tensorial": 0.9}[
            case.model
        ]
        error = model_factor / float(case.N_q or case.N_r)
        online = 0.2 + 0.001 * case.reduced_dynamical_dimension
        speedup = FOM_SECONDS / online
    metrics = {
        "metric_id": "relative_space_time_l2_error_v1",
        "relative_space_time_l2_error": error,
        "online_timing_id": (
            "rom_solve_ivp_only_v1" if case.execution_kind == "rom_integration" else None
        ),
        "fom_integration_elapsed_seconds": FOM_SECONDS,
        "rom_online_integration_elapsed_seconds": online,
        "online_speedup": speedup,
    }
    timings = {
        "lifting_construction_seconds": 0.01,
        "projected_operator_construction_seconds": 0.02,
        "inference_seconds": 0.03,
        "initial_coordinate_fitting_seconds": 0.04,
        "online_integration_seconds": online,
        "reconstruction_seconds": 0.05,
        "metric_evaluation_seconds": 0.06,
        "total_case_workflow_seconds": 0.5,
    }
    diagnostics = {
        "solver": {
            "success": case.execution_kind == "rom_integration",
            "message": "synthetic",
            "nfev": 11,
            "njev": 2,
            "nlu": 3,
            "final_time": 10.0 if case.execution_kind == "rom_integration" else None,
        },
        "inference": {
            "converged": True,
            "iteration_count": 0,
            "termination_reason": "synthetic",
        },
        "regularization": {
            "training_snapshot_count": TRAINING_COUNT,
            "catalog_coefficients": {
                "gamma": case.gamma,
                "lambda_L": case.lambda_L,
                "lambda_Q": case.lambda_Q,
            },
            "applied_gram_ridges": {
                "gamma": case.applied_lifting_ridge,
                "lambda_L": case.applied_linear_ridge,
                "lambda_Q": case.applied_quadratic_ridge,
            },
            "scaling_count": 1,
        },
        "stage_timing": timings,
        "projection_speedup_applicable": case.execution_kind != "projection_benchmark",
        "finite": True,
    }
    _write_json(root / "case.json", case.to_dict())
    _write_json(root / "metrics.json", {"schema_version": "1.0.0", "metrics": metrics})
    _write_json(
        root / "diagnostics.json",
        {"schema_version": "1.0.0", "diagnostics": diagnostics},
    )
    _write_json(
        root / "manifest.json",
        {
            "schema_version": "1.0.0",
            "status": "completed",
            "case_id": case.case_id,
            "case_definition_checksum_sha256": _case_checksum(case),
        },
    )


def _synthetic_results(output_root, run_id, *, omit=()):
    omitted = set(omit)
    for case in figure5_cases():
        if case.case_id not in omitted:
            _synthetic_case_result(output_root, run_id, case)
    phase_root = output_root / "phase7_runs" / run_id
    phase_root.mkdir(parents=True)
    _write_json(
        phase_root / "asset_validation.json",
        {
            "catalog_checksum_sha256": "synthetic-catalog",
            "fom_manifest": {
                "sha256": "synthetic-fom-manifest",
                "integration_elapsed_seconds": FOM_SECONDS,
            },
            "shared_offline_manifest": {"sha256": "synthetic-shared-manifest"},
            "environment": {
                "platform": "synthetic",
                "python_version": "synthetic",
                "numpy_version": np.__version__,
                "scipy_version": "synthetic",
                "blas": {"name": "synthetic"},
            },
        },
    )


def test_approved_metric_has_exact_trapezoids_endpoints_and_square_root():
    time = np.array([0.0, 2.0, 10.0])
    error_energy = np.array([4.0, 0.0, 0.0])
    reference_energy = np.ones(3)
    value = relative_space_time_l2_error_from_energies(
        error_energy, reference_energy, time
    )
    assert value == pytest.approx(np.sqrt(4.0 / 10.0))
    assert value != pytest.approx(np.mean(np.sqrt(error_energy / reference_energy)))
    assert value**2 == pytest.approx(4.0 / 10.0)


def test_approved_metric_uses_full_uncentered_fom_denominator():
    time = np.array([0.0, 5.0, 10.0])
    fom = np.full((1, 3), 2.0)
    rom = np.full((1, 3), 1.0)
    assert relative_space_time_l2_error(fom, rom, np.eye(1), time) == pytest.approx(
        0.5
    )
    with pytest.raises(ValueError, match=r"full \[0,10\] interval"):
        relative_space_time_l2_error(fom[:, :2], rom[:, :2], np.eye(1), time[:2])


def test_online_timer_brackets_only_solve_ivp(monkeypatch):
    model = NonlinearManifoldReducedModel(None)
    model.initial_condition = np.array([1.0])
    model.projectedLinear = np.array([[1.0]])
    model.TT = 10.0
    model.time_steps = np.array([0.0, 10.0])
    result = SimpleNamespace(
        success=True,
        message="ok",
        t=model.time_steps,
        y=np.array([[1.0, 0.0]]),
        nfev=4,
        njev=1,
        nlu=2,
    )
    monkeypatch.setattr(legacy_rom.sp.integrate, "solve_ivp", lambda **kwargs: result)
    clock = iter((100.0, 102.5))
    monkeypatch.setattr(
        legacy_rom, "time", SimpleNamespace(perf_counter=lambda: next(clock))
    )
    assert model.integrate_reduced(intrusive=True) is result
    assert model.last_solve_ivp_elapsed_seconds == 2.5


def test_figure5_plan_dimensions_scaling_reuse_and_scope(tmp_path):
    cases = figure5_cases()
    assert len(cases) == 58
    assert sum(case.execution_kind == "rom_integration" for case in cases) == 50
    assert sum(case.execution_kind == "projection_benchmark" for case in cases) == 8
    assert not any("fig4" in case.case_id for case in cases)
    assert len([case for case in cases if case.model == "linear" and case.N_r == 32]) == 2
    assert {
        case.reduced_dynamical_dimension
        for case in cases
        if case.model == "linear" and case.N_r != 32
    } == {32 + nq for nq in NQ_VALUES}
    elementwise = next(
        case
        for case in cases
        if case.operators == "inferred" and case.model == "elementwise"
    )
    tensorial = next(
        case
        for case in cases
        if case.operators == "inferred" and case.model == "tensorial"
    )
    assert elementwise.applied_lifting_ridge == pytest.approx(0.0060008)
    assert elementwise.applied_quadratic_ridge == pytest.approx(0.0060008)
    assert tensorial.applied_lifting_ridge == pytest.approx(0.000187525)
    assert tensorial.applied_quadratic_ridge == pytest.approx(0.0030004)

    completed = cases[0]
    _synthetic_case_result(tmp_path, "resume", completed)
    interrupted = cases[1]
    interrupted_root = tmp_path / "figure5_cases" / "resume" / interrupted.case_id
    interrupted_root.mkdir(parents=True)
    _write_json(
        interrupted_root / "manifest.json",
        {
            "status": "interrupted",
            "case_definition_checksum_sha256": _case_checksum(interrupted),
        },
    )
    plan = figure5_execution_plan(run_id="resume", output_root=tmp_path)
    by_id = {entry["case_id"]: entry for entry in plan["cases"]}
    assert by_id[completed.case_id]["reuse_status"] == "reused_completed_result"
    assert by_id[interrupted.case_id]["reuse_status"] == "resume_interrupted_case"
    assert plan["forbidden_scope"] == {
        "figure4_execution": False,
        "fom_execution": False,
        "derivative_recomputation": False,
        "pod_svd_recomputation": False,
        "regularization_search": False,
    }


def test_bundle_complete_partial_speedups_and_bundle_only_plot(tmp_path, monkeypatch):
    complete_root = tmp_path / "complete-results"
    _synthetic_results(complete_root, "complete")
    bundle = build_figure5_bundle(
        run_id="complete",
        output_directory=tmp_path / "complete-bundle",
        output_root=complete_root,
    )
    metadata, arrays = validate_figure5_bundle(bundle)
    assert metadata["case_set_status"] == "complete"
    assert metadata["complete_publication_reproduction"] is False
    assert metadata["dataset_checksum_sha256"] == EXPECTED_DATASET_CHECKSUM
    assert arrays["projected__fixed_linear__online_speedup"][0] == pytest.approx(
        FOM_SECONDS / 0.25
    )
    assert not any("best_projection__online_speedup" in name for name in arrays)
    required_timing_fields = {
        "lifting_construction_seconds",
        "projected_operator_construction_seconds",
        "inference_seconds",
        "initial_coordinate_fitting_seconds",
        "online_integration_seconds",
        "reconstruction_seconds",
        "metric_evaluation_seconds",
        "total_case_workflow_seconds",
    }
    integrated = next(
        record
        for record in metadata["case_records"].values()
        if record["case"]["execution_kind"] == "rom_integration"
    )
    assert required_timing_fields <= set(integrated["diagnostics"]["stage_timing"])

    monkeypatch.setattr(
        legacy_rom.sp.integrate,
        "solve_ivp",
        lambda **kwargs: pytest.fail("plotting invoked solve_ivp"),
    )
    outputs = plot_figure5_bundle(bundle, output_directory=tmp_path / "plot")
    assert {path.suffix for path in outputs} >= {".png", ".pdf", ".json", ".md"}
    plot_metadata = json.loads((tmp_path / "plot" / "plot_metadata.json").read_text())
    assert plot_metadata["layout"] == "manuscript_2_by_2"
    assert plot_metadata["projection_benchmark_in_speedup_panels"] is False
    assert plot_metadata["scientific_execution"]["solver_run"] is False
    assert plot_metadata["axis_scales"]["speedup"] == "logarithmic"
    assert plot_metadata["axis_ticks"]["speedup"] == [50, 100, 200, 300, 500]
    assert plot_metadata["titles"] == {
        "overall": "Effect of lifting dimension on accuracy and online efficiency",
        "panels": {
            "projected": "Projected streaming operator",
            "inferred": "Inferred streaming operator",
        },
    }
    assert plot_metadata["rendering_provenance"] == {
        "classification": "presentation_only_rerender",
        "source_bundle_validated_and_unchanged": True,
        "scientific_values_changed": False,
        "description": (
            "Presentation-only rerender from the unchanged validated Figure 5 bundle."
        ),
    }
    caption = (tmp_path / "plot" / "figure5_caption.md").read_text()
    assert "regenerated sigmoid benchmark" in caption
    assert "presentation-only rerender" in caption

    partial_root = tmp_path / "partial-results"
    omitted = figure5_cases()[-1].case_id
    _synthetic_results(partial_root, "partial", omit=(omitted,))
    partial_bundle = build_figure5_bundle(
        run_id="partial",
        output_directory=tmp_path / "partial-bundle",
        output_root=partial_root,
        allow_partial=True,
    )
    partial_metadata, _ = validate_figure5_bundle(
        partial_bundle, require_complete=False
    )
    assert partial_metadata["case_set_status"] == "partial"
    assert omitted in partial_metadata["missing_case_ids"]
    with pytest.raises(ValueError, match="case set is partial"):
        validate_figure5_bundle(partial_bundle)


def test_plot_script_has_no_solver_or_scientific_execution_calls():
    plotting_source = (REPOSITORY_ROOT / "one_d" / "figure5_plotting.py").read_text()
    tree = ast.parse(plotting_source)
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {"solve_ivp", "execute_figure5_plan", "assemble_operators"} & called_names


def test_independent_golden_checksum_is_unchanged():
    manifest = json.loads(
        (REPOSITORY_ROOT / "tests" / "golden" / "tiny_1d_manifest.json").read_text()
    )
    assert manifest["content_checksum"]["sha256"] == GOLDEN_CHECKSUM
