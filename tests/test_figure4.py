import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from one_d.figure4 import (
    GAMMA_GRID,
    LAMBDA_Q_GRID,
    MODELS,
    OPERATORS,
    RANKS,
    RESULT_LABEL,
    SELECTION_PROVENANCE,
    TOTAL_NONLINEAR_DIMENSION,
    _coarse_candidate,
    _execute_candidate,
    _linear_case,
    _write_final_result,
    build_figure4_bundle,
    canonical_checksum,
    geometric_refinement_values,
    search_definition,
    select_candidate,
    validate_candidate_result,
    validate_search_definition,
    validate_selected_parameters,
    write_selected_parameters,
)
from one_d.figure4_plotting import plot_figure4_bundle, validate_figure4_bundle
from one_d.figure5 import TRAINING_COUNT


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FOM_SECONDS = 662.6050200000172
GOLDEN_CHECKSUM = "91c84e813e5cbfabd0bf0c5be436afc19e64152b7f06c9f1a572a76038108238"


def _write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _definition(run_id="synthetic"):
    return search_definition(
        run_id=run_id,
        source_state={
            "algorithm": "synthetic",
            "sha256": "source-state",
            "file_count": 1,
            "git_commit": "commit",
        },
        catalog_checksum_sha256="synthetic-catalog",
    )


def _candidate_record(candidate, error, *, admissible=True):
    return {
        "candidate": candidate.to_dict(),
        "admissible": admissible,
        "relative_space_time_l2_error": error,
        "artifact_path": f"/synthetic/{candidate.candidate_id}",
    }


def _runner(*, error=0.25, inferred=False, converged=True, finite=True):
    metrics = {
        "metric_id": "relative_space_time_l2_error_v1",
        "relative_space_time_l2_error": error,
        "online_timing_id": "rom_solve_ivp_only_v1",
        "fom_integration_elapsed_seconds": FOM_SECONDS,
        "rom_online_integration_elapsed_seconds": 0.5,
        "online_speedup": FOM_SECONDS / 0.5,
    }
    diagnostics = {
        "solver": {
            "success": True,
            "message": "synthetic",
            "nfev": 4,
            "njev": 0,
            "nlu": 0,
            "final_time": 10.0,
        },
        "inference": {"converged": converged} if inferred else {},
        "reduced_initial_condition": {
            "finite": True,
            "POD_coefficient_fit_residual": 0.0,
        },
        "finite": finite,
    }
    return metrics, diagnostics


def _execution():
    return SimpleNamespace(catalog=SimpleNamespace(checksum=lambda: "synthetic-catalog"))


def _synthetic_selections():
    selections = {}
    for operators in OPERATORS:
        for model in MODELS:
            for rank in RANKS:
                gamma = GAMMA_GRID[(rank // 8) % len(GAMMA_GRID)]
                coefficient = gamma if operators == "projected" else LAMBDA_Q_GRID[3]
                candidate = _coarse_candidate(
                    model=model,
                    operators=operators,
                    rank=rank,
                    coefficient=coefficient,
                    grid_index=3,
                    gamma=gamma if operators == "inferred" else None,
                )
                record = _candidate_record(candidate, 0.5 / rank)
                selections[f"{operators}_{model}_nr{rank}"] = {
                    "selected_record": record,
                    "final_decision": {
                        "candidate_rank": 1,
                        "tied_candidate_ids": [candidate.candidate_id],
                        "larger_regularization_tie_choice_applied": False,
                    },
                }
    return selections


def _synthetic_figure4_results(output_root, run_id, *, omitted=()):
    omitted = set(omitted)
    phase_root = output_root / "phase8_runs" / run_id
    phase_root.mkdir(parents=True)
    definition = _definition(run_id)
    _write_json(phase_root / "search_definition.json", definition)
    selections = _synthetic_selections()
    selected_path, _ = write_selected_parameters(
        phase_root=phase_root,
        run_id=run_id,
        definition=definition,
        selections=selections,
    )
    selected = validate_selected_parameters(selected_path, expected_run_id=run_id)
    execution = _execution()
    case_count = 0
    for operators in OPERATORS:
        for rank in RANKS:
            linear = _linear_case(rank, operators)
            if linear.case_id not in omitted:
                _write_final_result(
                    root=output_root / "figure4_cases" / run_id / linear.case_id,
                    case=linear,
                    metrics=_runner(error=0.8 / rank)[0],
                    diagnostics=_runner()[1],
                    definition=definition,
                    execution=execution,
                    selected_candidate=None,
                )
                case_count += 1
    for case_id, values in selected["cases"].items():
        if case_id in omitted:
            continue
        candidate = next(
            selection["selected_record"]["candidate"]
            for selection in selections.values()
            if Path(values["candidate_artifact_path"]).name
            == selection["selected_record"]["candidate"]["candidate_id"]
        )
        case = _coarse_candidate(
            model=candidate["model"],
            operators=candidate["operators"],
            rank=candidate["N_r"],
            coefficient=(
                candidate["gamma"]
                if candidate["operators"] == "projected"
                else candidate["lambda_Q"]
            ),
            grid_index=3,
            gamma=candidate["gamma"] if candidate["operators"] == "inferred" else None,
        ).rom_case()
        case = type(case)(**{**case.to_dict(), "case_id": case_id})
        _write_final_result(
            root=output_root / "figure4_cases" / run_id / case_id,
            case=case,
            metrics=_runner(error=(0.65 if case.model == "elementwise" else 0.5) / case.N_r)[0],
            diagnostics=_runner(inferred=case.operators == "inferred")[1],
            definition=definition,
            execution=execution,
            selected_candidate={"candidate_id": candidate["candidate_id"]},
        )
        case_count += 1
    _write_json(
        phase_root / "search_index.json",
        {
            "schema_version": "1.0.0",
            "status": "complete" if not omitted else "partial",
            "run_id": run_id,
            "completed_final_cases": case_count,
            "expected_final_cases": 48,
        },
    )


def test_candidate_grids_are_exact_geometric_nine_point_ranges():
    assert len(GAMMA_GRID) == len(LAMBDA_Q_GRID) == 9
    assert GAMMA_GRID[0] == 7.0e-10
    assert GAMMA_GRID[-1] == 5.0e-5
    assert LAMBDA_Q_GRID[0] == 6.0e-9
    assert LAMBDA_Q_GRID[-1] == 2.0e-4
    assert np.allclose(
        np.asarray(GAMMA_GRID[1:]) / np.asarray(GAMMA_GRID[:-1]),
        (GAMMA_GRID[-1] / GAMMA_GRID[0]) ** (1.0 / 8.0),
    )
    assert np.allclose(
        np.asarray(LAMBDA_Q_GRID[1:]) / np.asarray(LAMBDA_Q_GRID[:-1]),
        (LAMBDA_Q_GRID[-1] / LAMBDA_Q_GRID[0]) ** (1.0 / 8.0),
    )


def test_dimensions_and_coefficient_scaling_are_applied_exactly_once():
    projected = _coarse_candidate(
        model="elementwise",
        operators="projected",
        rank=40,
        coefficient=GAMMA_GRID[4],
        grid_index=4,
    )
    inferred = _coarse_candidate(
        model="tensorial",
        operators="inferred",
        rank=56,
        coefficient=LAMBDA_Q_GRID[5],
        grid_index=5,
        gamma=GAMMA_GRID[2],
    )
    assert projected.N_r + projected.N_q == TOTAL_NONLINEAR_DIMENSION
    assert inferred.N_r + inferred.N_q == TOTAL_NONLINEAR_DIMENSION
    assert projected.applied_gamma_ridge == projected.gamma * TRAINING_COUNT
    assert inferred.applied_gamma_ridge == inferred.gamma * TRAINING_COUNT
    assert inferred.applied_lambda_Q_ridge == inferred.lambda_Q * TRAINING_COUNT
    for rank in RANKS:
        assert _linear_case(rank, "projected").N_q is None


def test_selection_objective_tie_rule_and_larger_regularization_choice():
    candidates = [
        _coarse_candidate(
            model="elementwise",
            operators="projected",
            rank=8,
            coefficient=value,
            grid_index=index,
        )
        for index, value in enumerate(GAMMA_GRID[:3])
    ]
    records = [
        _candidate_record(candidates[0], 0.10000),
        _candidate_record(candidates[1], 0.10005),
        _candidate_record(candidates[2], 0.10011),
    ]
    decision = select_candidate(records, coefficient_field="gamma")
    assert decision["selected_candidate_id"] == candidates[1].candidate_id
    assert decision["larger_regularization_tie_choice_applied"]
    assert decision["tied_candidate_ids"] == [
        candidates[0].candidate_id,
        candidates[1].candidate_id,
    ]
    lambda_candidate = _coarse_candidate(
        model="tensorial",
        operators="inferred",
        rank=8,
        coefficient=LAMBDA_Q_GRID[0],
        grid_index=0,
        gamma=GAMMA_GRID[0],
    )
    assert select_candidate(
        [_candidate_record(lambda_candidate, 0.2)], coefficient_field="lambda_Q"
    )["selected_coefficient"] == LAMBDA_Q_GRID[0]


def test_geometric_midpoint_refinement_and_endpoint_behavior():
    interior = geometric_refinement_values(GAMMA_GRID, 4)
    assert [neighbor for neighbor, _ in interior] == ["lower", "upper"]
    assert interior[0][1] == pytest.approx(np.sqrt(GAMMA_GRID[3] * GAMMA_GRID[4]))
    assert interior[1][1] == pytest.approx(np.sqrt(GAMMA_GRID[4] * GAMMA_GRID[5]))
    assert geometric_refinement_values(GAMMA_GRID, 0) == [
        ("upper", pytest.approx(np.sqrt(GAMMA_GRID[0] * GAMMA_GRID[1])))
    ]
    assert geometric_refinement_values(GAMMA_GRID, 8) == [
        ("lower", pytest.approx(np.sqrt(GAMMA_GRID[7] * GAMMA_GRID[8])))
    ]


def test_candidate_admissibility_rejection_and_resume(tmp_path):
    definition = _definition("candidate")
    projected = _coarse_candidate(
        model="elementwise",
        operators="projected",
        rank=8,
        coefficient=GAMMA_GRID[0],
        grid_index=0,
    )
    calls = []

    def runner():
        calls.append(True)
        return _runner()

    root = tmp_path / projected.candidate_id
    first = _execute_candidate(
        candidate=projected,
        root=root,
        definition=definition,
        execution=_execution(),
        runner=runner,
    )
    second = _execute_candidate(
        candidate=projected,
        root=root,
        definition=definition,
        execution=_execution(),
        runner=lambda: pytest.fail("completed candidate was rerun"),
    )
    assert first["admissible"] and second["admissible"]
    assert len(calls) == 1
    validate_candidate_result(root, projected)

    unstable = _coarse_candidate(
        model="tensorial",
        operators="projected",
        rank=8,
        coefficient=GAMMA_GRID[1],
        grid_index=1,
    )
    rejected = _execute_candidate(
        candidate=unstable,
        root=tmp_path / unstable.candidate_id,
        definition=definition,
        execution=_execution(),
        runner=lambda: _runner(error=np.nan, finite=False),
    )
    assert not rejected["admissible"]

    inferred = _coarse_candidate(
        model="elementwise",
        operators="inferred",
        rank=8,
        coefficient=LAMBDA_Q_GRID[0],
        grid_index=0,
        gamma=GAMMA_GRID[0],
    )
    nonconverged = _execute_candidate(
        candidate=inferred,
        root=tmp_path / inferred.candidate_id,
        definition=definition,
        execution=_execution(),
        runner=lambda: _runner(inferred=True, converged=False),
    )
    assert not nonconverged["admissible"]


def test_search_definition_is_deterministic_and_records_scope():
    first = _definition("deterministic")
    second = _definition("deterministic")
    assert first == second
    validate_search_definition(first)
    payload = dict(first)
    checksum = payload.pop("content_checksum_sha256")
    assert checksum == canonical_checksum(payload)
    assert first["planned_counts"] == {
        "projected_coarse": 144,
        "projected_refinement_maximum": 32,
        "inferred_coarse": 144,
        "inferred_refinement_maximum": 32,
        "nonlinear_candidate_maximum": 352,
        "linear_final_cases": 16,
        "nonlinear_final_cases": 32,
    }
    assert first["scientific_execution"] == {
        "fom_run": False,
        "derivatives_recomputed": False,
        "pod_svd_recomputed": False,
        "figure5_rerun": False,
        "search_outside_approved_ranges": False,
    }


def test_selected_parameter_serialization_checksum_and_candidate_reuse(tmp_path):
    selections = _synthetic_selections()
    path, proposed = write_selected_parameters(
        phase_root=tmp_path,
        run_id="selected",
        definition=_definition("selected"),
        selections=selections,
    )
    selected = validate_selected_parameters(path, expected_run_id="selected")
    assert len(selected["cases"]) == 32
    assert proposed.is_file()
    assert all(
        value["provenance_status"] == SELECTION_PROVENANCE
        for value in selected["cases"].values()
    )
    assert all(
        value["candidate_artifact_path"].endswith(
            next(
                selection["selected_record"]["candidate"]["candidate_id"]
                for selection in selections.values()
                if selection["selected_record"]["artifact_path"]
                == value["candidate_artifact_path"]
            )
        )
        for value in selected["cases"].values()
    )
    original = path.read_bytes()
    write_selected_parameters(
        phase_root=tmp_path,
        run_id="selected",
        definition=_definition("selected"),
        selections=selections,
    )
    assert path.read_bytes() == original


def test_complete_and_partial_48_case_bundles_and_bundle_only_plotting(tmp_path):
    _synthetic_figure4_results(tmp_path, "complete")
    bundle = build_figure4_bundle(
        run_id="complete",
        output_directory=tmp_path / "bundle",
        output_root=tmp_path,
    )
    metadata, arrays = validate_figure4_bundle(bundle)
    assert metadata["case_set_status"] == "complete"
    assert metadata["result_label"] == RESULT_LABEL
    assert len(metadata["case_records"]) == 48
    assert len(arrays) == 26
    outputs = plot_figure4_bundle(bundle, output_directory=tmp_path / "plot")
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".json", ".md"}
    plot_metadata = json.loads((tmp_path / "plot" / "plot_metadata.json").read_text())
    assert plot_metadata["scientific_execution"]["solver_run"] is False
    assert plot_metadata["selected_parameter_checksum_sha256"] == metadata[
        "selected_parameters"
    ]["content_checksum_sha256"]

    omitted = "fig4_tensorial_inferred_nr64"
    _synthetic_figure4_results(tmp_path, "partial", omitted=(omitted,))
    partial = build_figure4_bundle(
        run_id="partial",
        output_directory=tmp_path / "partial-bundle",
        output_root=tmp_path,
        allow_partial=True,
    )
    partial_metadata, _ = validate_figure4_bundle(partial, require_complete=False)
    assert partial_metadata["case_set_status"] == "partial"
    assert omitted in partial_metadata["missing_case_ids"]
    with pytest.raises(ValueError, match="partial"):
        validate_figure4_bundle(partial)


def test_plotting_has_no_solver_import_or_invocation():
    path = REPOSITORY_ROOT / "one_d" / "figure4_plotting.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name.startswith("one_d") or name.endswith("figure4") for name in imports)
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not {"solve_ivp", "execute_phase8", "run_fom", "compute_derivatives"} & called


def test_execution_source_forbids_large_scientific_recomputation():
    source = (REPOSITORY_ROOT / "one_d" / "figure4.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "svd" not in called_attributes
    assert not {"run_fom", "compute_derivatives", "execute_figure5_plan"} & called_names
    assert "full_reconstructed_trajectory_constructed" in source
    assert "shared_metric_inputs_path" in source


def test_independent_golden_checksum_is_unchanged():
    metadata = json.loads(
        (REPOSITORY_ROOT / "tests" / "golden" / "tiny_1d_manifest.json").read_text()
    )
    assert metadata["content_checksum"]["sha256"] == GOLDEN_CHECKSUM
    with np.load(
        REPOSITORY_ROOT / "tests" / "golden" / "tiny_1d_reference.npz",
        allow_pickle=False,
    ) as archive:
        digest = hashlib.sha256()
        for name in sorted(archive.files):
            array = np.ascontiguousarray(archive[name])
            for field in (
                name.encode("utf-8"),
                array.dtype.str.encode("ascii"),
                json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"),
                array.tobytes(order="C"),
            ):
                digest.update(len(field).to_bytes(8, "big"))
                digest.update(field)
    assert digest.hexdigest() == GOLDEN_CHECKSUM
