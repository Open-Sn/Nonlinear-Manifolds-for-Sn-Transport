import hashlib
import json
from pathlib import Path
import shutil

import pytest

from one_d.publication_experiments import (
    dry_run_publication_case,
    load_figure4_selected_parameters,
    load_publication_catalog,
    resolve_case_configuration,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLICATION_CONFIG_ROOT = REPOSITORY_ROOT / "configs" / "1d" / "publication"
CATALOG_PATH = PUBLICATION_CONFIG_ROOT / "experiments.json"
SELECTED_PATH = PUBLICATION_CONFIG_ROOT / "figure4_selected_parameters.json"
SCHEMA_PATH = PUBLICATION_CONFIG_ROOT / "figure4_selected_parameters.schema.json"
LEGACY_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "1d" / "legacy_production.json"
TRACKED_CONTENT_CHECKSUM = "afd75f91a16b8dc3b87484752f2b699203d980ea577fa2721124ac6ae9d43d4e"
CATALOG_CHECKSUM = "b53b7482b846fbc2bad3610b89568c91cb682d6d21a3ec265f34d308608e9b73"

EXPECTED_VALUES = {
    "fig4_elementwise_inferred_nr8": (5e-05, 0.0, 0.0001043157954515697),
    "fig4_elementwise_inferred_nr16": (6.149913551867406e-06, 0.0, 5.4408925903468654e-05),
    "fig4_elementwise_inferred_nr24": (5e-05, 0.0, 4.026711047205203e-06),
    "fig4_elementwise_inferred_nr32": (7.564287339088475e-07, 0.0, 4.026711047205203e-06),
    "fig4_elementwise_inferred_nr40": (1.8708286933869707e-07, 0.0, 1.4801656089845709e-05),
    "fig4_elementwise_inferred_nr48": (1.8708286933869707e-07, 0.0, 5.713611427291954e-07),
    "fig4_elementwise_inferred_nr56": (1.8708286933869707e-07, 0.0, 5.713611427291954e-07),
    "fig4_elementwise_inferred_nr64": (2.301086946936585e-08, 0.0, 2.9800996046956936e-07),
    "fig4_elementwise_projected_nr8": (5e-05, None, None),
    "fig4_elementwise_projected_nr16": (6.149913551867406e-06, None, None),
    "fig4_elementwise_projected_nr24": (5e-05, None, None),
    "fig4_elementwise_projected_nr32": (7.564287339088475e-07, None, None),
    "fig4_elementwise_projected_nr40": (1.8708286933869707e-07, None, None),
    "fig4_elementwise_projected_nr48": (1.8708286933869707e-07, None, None),
    "fig4_elementwise_projected_nr56": (1.8708286933869707e-07, None, None),
    "fig4_elementwise_projected_nr64": (2.301086946936585e-08, None, None),
    "fig4_tensorial_inferred_nr8": (1.521020318097722e-06, 0.0, 7.72023264506413e-06),
    "fig4_tensorial_inferred_nr16": (2.301086946936585e-08, 0.0, 5.713611427291954e-07),
    "fig4_tensorial_inferred_nr24": (4.627005616132191e-08, 0.0, 5.713611427291954e-07),
    "fig4_tensorial_inferred_nr32": (2.301086946936585e-08, 0.0, 2.9800996046956936e-07),
    "fig4_tensorial_inferred_nr40": (2.8302971597981057e-09, 0.0, 2.20552047310954e-08),
    "fig4_tensorial_inferred_nr48": (1.4075539108178678e-09, 0.0, 1.1503531126857196e-08),
    "fig4_tensorial_inferred_nr56": (1.4075539108178678e-09, 0.0, 1.1503531126857196e-08),
    "fig4_tensorial_inferred_nr64": (7e-10, 0.0, 6e-09),
    "fig4_tensorial_projected_nr8": (1.521020318097722e-06, None, None),
    "fig4_tensorial_projected_nr16": (2.301086946936585e-08, None, None),
    "fig4_tensorial_projected_nr24": (4.627005616132191e-08, None, None),
    "fig4_tensorial_projected_nr32": (2.301086946936585e-08, None, None),
    "fig4_tensorial_projected_nr40": (2.8302971597981057e-09, None, None),
    "fig4_tensorial_projected_nr48": (1.4075539108178678e-09, None, None),
    "fig4_tensorial_projected_nr56": (1.4075539108178678e-09, None, None),
    "fig4_tensorial_projected_nr64": (7e-10, None, None),
}


def _canonical_checksum(value):
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _copy_portable_inputs(root):
    legacy = root / "configs" / "1d" / "legacy_production.json"
    selected = root / "configs" / "1d" / "publication" / SELECTED_PATH.name
    legacy.parent.mkdir(parents=True)
    selected.parent.mkdir(parents=True)
    shutil.copy2(LEGACY_CONFIG_PATH, legacy)
    shutil.copy2(SELECTED_PATH, selected)


def test_selected_parameter_schema_contract_and_exact_phase8_values():
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    selected = load_figure4_selected_parameters(SELECTED_PATH)
    assert schema["$schema"].endswith("2020-12/schema")
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(selected)
    assert schema["properties"]["cases"]["minProperties"] == 32
    assert schema["properties"]["cases"]["maxProperties"] == 32
    assert selected["content_checksum_sha256"] == TRACKED_CONTENT_CHECKSUM
    payload = dict(selected)
    assert payload.pop("content_checksum_sha256") == _canonical_checksum(payload)
    actual = {
        case_id: (case["gamma"], case.get("lambda_L"), case.get("lambda_Q"))
        for case_id, case in selected["cases"].items()
    }
    assert actual == EXPECTED_VALUES


def test_selected_parameter_dimensions_scaling_reuse_uniqueness_and_provenance():
    selected = load_figure4_selected_parameters(SELECTED_PATH)
    assert selected["parameter_provenance"] == "regenerated_sigmoid_search"
    assert selected["historical_parameter_recovery"] is False
    assert selected["complete_publication_reproduction"] is False
    assert selected["training_snapshot_count"] == 7501
    combinations = set()
    for case_id, case in selected["cases"].items():
        combination = (case["model"], case["operator_type"], case["N_r"])
        assert combination not in combinations
        combinations.add(combination)
        assert case["N_r"] + case["N_q"] == 564
        assert case["origin"] in {"coarse", "refined"}
        assert case["applied_ridges"]["gamma"] == case["gamma"] * 7501
        if case["operator_type"] == "projected":
            assert "lambda_L" not in case and "lambda_Q" not in case
            assert set(case["applied_ridges"]) == {"gamma"}
        else:
            assert case["lambda_L"] == 0.0
            assert case["applied_ridges"]["lambda_Q"] == case["lambda_Q"] * 7501
            projected = selected["cases"][case["gamma_source_case_id"]]
            assert case["gamma"] == projected["gamma"]
    assert len(combinations) == 32


def test_selected_parameter_provenance_contains_no_absolute_paths():
    selected = load_figure4_selected_parameters(SELECTED_PATH)
    serialized = json.dumps(selected, sort_keys=True)
    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    hint = selected["relative_result_root_hint"]
    assert not Path(hint).is_absolute()
    assert hint == "results/1d/publication/phase8_runs/phase8-20260802T130106Z"


def test_catalog_links_all_32_regenerated_cases_and_preserves_other_figures():
    catalog = load_publication_catalog()
    assert catalog.checksum() == CATALOG_CHECKSUM
    explicit_cases_checksum = _canonical_checksum(catalog.source_data["cases"])
    assert explicit_cases_checksum == "ae7819ee79f7d26275903925b1a612bc550d80cff58b1cfe2ea044e5b308a7f5"
    figure4 = [case for case in catalog.cases if case.figure == "Figure 4"]
    assert len(figure4) == 48
    assert all(case.fully_specified and case.execution_allowed for case in figure4)
    nonlinear = [case for case in figure4 if case.model_type != "linear"]
    linear = [case for case in figure4 if case.model_type == "linear"]
    assert len(nonlinear) == 32 and len(linear) == 16
    assert all(
        case.provenance["parameter_provenance"] == "regenerated_sigmoid_search"
        and case.provenance["selected_parameter_file"]
        == "configs/1d/publication/figure4_selected_parameters.json"
        for case in nonlinear
    )
    assert all(
        case.provenance["parameter_provenance"] == "catalog_direct_linear"
        for case in linear
    )
    manifest = json.loads(
        (REPOSITORY_ROOT / "tests" / "golden" / "tiny_1d_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["content_checksum"]["sha256"] == (
        "91c84e813e5cbfabd0bf0c5be436afc19e64152b7f06c9f1a572a76038108238"
    )


def test_catalog_resolution_is_portable_without_phase8_results(tmp_path):
    _copy_portable_inputs(tmp_path)
    assert not (tmp_path / "results").exists()
    catalog = load_publication_catalog(CATALOG_PATH, repository_root=tmp_path)
    case = catalog.get("fig4_tensorial_inferred_nr32")
    config = resolve_case_configuration(case, repository_root=tmp_path)
    report = dry_run_publication_case(
        catalog,
        case,
        snapshot_path=tmp_path / "missing.npy",
        repository_root=tmp_path,
    )
    assert config.rom.lifting_regularization == 2.301086946936585e-08
    assert config.rom.quadratic_inference_regularization_tensorial == 2.9800996046956936e-07
    assert report.action == "refuse_missing_snapshot"
    assert report.N_s == 7501
    assert report.parameter_provenance == "regenerated_sigmoid_search"
    assert report.assembles_operators is False
    assert report.solves is False
    assert report.writes_files is False
    assert not (tmp_path / "results").exists()


def test_missing_selected_parameter_file_has_clear_error(tmp_path):
    legacy = tmp_path / "configs" / "1d" / "legacy_production.json"
    legacy.parent.mkdir(parents=True)
    shutil.copy2(LEGACY_CONFIG_PATH, legacy)
    with pytest.raises(FileNotFoundError, match="tracked Figure 4 selected-parameter file is missing"):
        load_publication_catalog(CATALOG_PATH, repository_root=tmp_path)


def test_tampered_selected_parameter_checksum_has_clear_error(tmp_path):
    tampered = tmp_path / SELECTED_PATH.name
    value = json.loads(SELECTED_PATH.read_text(encoding="utf-8"))
    value["cases"]["fig4_tensorial_inferred_nr32"]["gamma"] = 1e-8
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="selected-parameter checksum mismatch"):
        load_figure4_selected_parameters(tampered)


def test_generic_resolution_does_not_invoke_phase8_search(monkeypatch):
    import one_d.figure4 as figure4_search

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Phase 8 candidate search must not run during catalog resolution")

    monkeypatch.setattr(figure4_search, "execute_phase8", fail_if_called)
    catalog = load_publication_catalog()
    case = catalog.get("fig4_tensorial_inferred_nr32")
    config = resolve_case_configuration(case)
    assert config.rom.lifting_regularization == 2.301086946936585e-08
    assert config.rom.quadratic_inference_regularization_tensorial == 2.9800996046956936e-07
