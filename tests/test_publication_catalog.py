import json
from pathlib import Path

import numpy as np
import pytest

from one_d.publication_experiments import (
    BENCHMARK_VARIANT,
    CANONICAL_SNAPSHOT_FILENAME,
    EXPECTED_DEVIATION,
    LEGACY_CONFIG_CHECKSUM,
    SIGMOID_FORMULA,
    dry_run_publication_case,
    load_publication_catalog,
    resolve_base_configuration,
)
from one_d.publication_metrics import (
    instantaneous_error_history,
    pod_energy_curves,
    publication_convergence_metric,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPOSITORY_ROOT / "configs" / "1d" / "publication" / "experiments.json"


@pytest.fixture(scope="module")
def catalog():
    return load_publication_catalog()


def test_catalog_is_deterministic_and_has_pinned_checksum(catalog):
    reconstructed = load_publication_catalog(CATALOG_PATH)
    assert catalog.canonical_json() == reconstructed.canonical_json()
    assert catalog.checksum() == reconstructed.checksum()
    assert catalog.checksum() == "b53b7482b846fbc2bad3610b89568c91cb682d6d21a3ec265f34d308608e9b73"
    assert len(catalog.cases) == 57
    assert sum(case.fully_specified for case in catalog.cases) == 57
    assert sum(case.specification_status == "partially_specified" for case in catalog.cases) == 0


def test_every_publication_case_resolves_to_exact_legacy_sigmoid(catalog):
    for case in catalog.cases:
        config = resolve_base_configuration(case)
        initial = config.problem.initial_condition
        assert case.base_configuration_checksum == LEGACY_CONFIG_CHECKSUM
        assert config.checksum() == LEGACY_CONFIG_CHECKSUM
        assert case.benchmark_variant == BENCHMARK_VARIANT
        assert case.manuscript_deviation == EXPECTED_DEVIATION
        assert case.required_input_snapshot == CANONICAL_SNAPSHOT_FILENAME
        assert initial.kind == "localized_sigmoid"
        assert initial.transition_location == 0.1
        assert initial.steepness == 100.0
        assert initial.amplitude == 1.0
        assert initial.angular_block == "final"
        report = dry_run_publication_case(catalog, case)
        assert report.sigmoid_initial_condition["formula"] == SIGMOID_FORMULA
        assert report.assembles_operators is False
        assert report.solves is False
        assert report.writes_files is False


@pytest.mark.parametrize(
    "override",
    [
        {"initial_condition": {"type": "zero"}},
        {"configuration_overrides": {"problem.initial_condition.kind": "zero"}},
        {"configuration_overrides": {"initial-condition": "localized_sigmoid"}},
    ],
)
def test_catalog_rejects_every_initial_condition_override(tmp_path, override):
    data = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    data["cases"][0].update(override)
    path = tmp_path / "experiments.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="may not override"):
        load_publication_catalog(path)


def test_figure1_study_values(catalog):
    case = catalog.get("fig1_pod_reducibility")
    assert case.latent_dimension == 16
    assert case.lifting_dimension == 548
    assert case.parameter_sweep["maximum_total_basis_dimension"] == 564
    assert case.training_interval == (0.0, 7.5)
    assert case.steady_state_centering
    assert case.fully_specified and case.execution_allowed


def test_figure2_dimensions_and_stage_specific_gamma(catalog):
    linear = catalog.get("fig2_linear_projected")
    elementwise = catalog.get("fig2_elementwise_projected")
    tensorial = catalog.get("fig2_tensorial_projected")
    assert linear.latent_dimension == 16 and linear.lifting_dimension is None
    assert linear.lifting_regularization_gamma is None
    assert elementwise.lifting_dimension == tensorial.lifting_dimension == 548
    assert elementwise.lifting_regularization_gamma == 6.4e-6
    assert tensorial.lifting_regularization_gamma == 2.5e-6
    for case in (linear, elementwise, tensorial):
        assert case.operator_construction == "projected"
        assert case.lambda_L is None and case.lambda_Q is None
        assert case.fully_specified


def test_figure3_inference_values_and_iteration_metadata(catalog):
    linear = catalog.get("fig3_linear_inferred")
    elementwise = catalog.get("fig3_elementwise_inferred")
    tensorial = catalog.get("fig3_tensorial_inferred")
    assert linear.lambda_L == 0.0 and linear.lambda_Q is None
    assert elementwise.lifting_regularization_gamma == 6.4e-6
    assert elementwise.lambda_L == 0.0 and elementwise.lambda_Q == 2.56e-5
    assert tensorial.lifting_regularization_gamma == 2.5e-6
    assert tensorial.lambda_L == 0.0 and tensorial.lambda_Q == 1.6e-6
    for case in (linear, elementwise, tensorial):
        assert case.inference_tolerance == 1.0e-6
        assert case.maximum_iterations == 100000
        assert case.fully_specified
    assert elementwise.reported_manuscript_metadata["reported_inference_iterations"] == 23367
    assert tensorial.reported_manuscript_metadata["reported_inference_iterations"] == 868
    assert elementwise.reported_manuscript_metadata["iteration_count_role"] == "manuscript_metadata_only"
    assert tensorial.reported_manuscript_metadata["iteration_count_role"] == "manuscript_metadata_only"


def test_figure4_expansion_constraint_and_readiness(catalog):
    cases = [case for case in catalog.cases if case.figure == "Figure 4"]
    ranks = {8, 16, 24, 32, 40, 48, 56, 64}
    expected_combinations = {
        (model, operators)
        for model in ("linear", "elementwise", "tensorial")
        for operators in ("projected", "inferred")
    }
    assert len(cases) == 48
    assert {case.latent_dimension for case in cases} == {8, 16, 24, 32, 40, 48, 56, 64}
    assert {case.model_type for case in cases} == {"linear", "elementwise", "tensorial"}
    assert {case.operator_construction for case in cases} == {"projected", "inferred"}
    for rank in ranks:
        assert {
            (case.model_type, case.operator_construction)
            for case in cases
            if case.latent_dimension == rank
        } == expected_combinations
    linear_cases = [case for case in cases if case.model_type == "linear"]
    nonlinear_cases = [case for case in cases if case.model_type != "linear"]
    assert len(linear_cases) == 16
    assert len(nonlinear_cases) == 32
    for case in linear_cases:
        assert case.lifting_dimension is None
        assert case.lifting_regularization_gamma is None
        assert case.lambda_Q is None
        assert case.inference_tolerance is None
        assert case.maximum_iterations is None
        assert case.specification_status == "fully_specified"
        assert case.execution_allowed
        assert case.missing_information == ()
        assert not any("gamma" in item or "lambda_Q" in item for item in case.missing_information)
        assert case.parameter_sweep["model_dimension_constraint"] == "linear model dimension = N_r"
        assert case.parameter_sweep["nonlinear_total_dimension_constraint_applicable"] is False
        if case.operator_construction == "projected":
            assert case.lambda_L is None
        else:
            assert case.lambda_L == 0.0
    for case in nonlinear_cases:
        assert case.latent_dimension + case.lifting_dimension == 564
        assert case.lifting_regularization_gamma is not None
        assert case.specification_status == "fully_specified"
        assert case.execution_allowed
        assert case.missing_information == ()
        assert case.provenance["parameter_provenance"] == "regenerated_sigmoid_search"
        assert case.provenance["historical_parameter_recovery"] is False
        assert case.provenance["complete_publication_reproduction"] is False
        if case.operator_construction == "projected":
            assert case.lambda_L is None
            assert case.lambda_Q is None
        else:
            assert case.lambda_L == 0.0
            assert case.lambda_Q is not None
    for case in cases:
        assert case.benchmark_variant == BENCHMARK_VARIANT
        assert case.manuscript_deviation == EXPECTED_DEVIATION
    selected = catalog.get("fig4_tensorial_inferred_nr32")
    assert selected.lifting_dimension == 532
    assert selected.parameter_sweep["gamma_candidate_range"] == [7e-10, 5e-5]
    assert selected.parameter_sweep["lambda_Q_candidate_range"] == [6e-9, 2e-4]


def test_figure5_series_dimensions_regularization_and_readiness(catalog):
    projected = catalog.get("fig5_projected_nq_sweep")
    inferred = catalog.get("fig5_inferred_nq_sweep")
    expected_nq = [1, 2, 4, 8, 16, 32, 64, 128]
    for case in (projected, inferred):
        assert case.latent_dimension == 32
        assert case.parameter_sweep["lifting_dimensions"] == expected_nq
        assert set(case.parameter_sweep["series"]) == {
            "fixed_rank_linear",
            "elementwise_quadratic",
            "tensorial_quadratic",
            "enlarged_linear",
            "M_orthogonal_best_projection",
        }
        assert case.specification_status == "fully_specified"
        assert case.execution_allowed
        assert case.missing_information == ()
        assert case.provenance["metric_id"] == "relative_space_time_l2_error_v1"
        assert case.provenance["online_timing_id"] == "rom_solve_ivp_only_v1"
    projected_series = projected.parameter_sweep["series"]
    assert projected_series["elementwise_quadratic"]["lifting_regularization_gamma"] == 8e-7
    assert projected_series["tensorial_quadratic"]["lifting_regularization_gamma"] == 2.5e-8
    assert all("inference_regularization" not in value for value in projected_series.values())
    inferred_series = inferred.parameter_sweep["series"]
    assert inferred_series["elementwise_quadratic"]["inference_regularization"] == {
        "lambda_L": 0.0,
        "lambda_Q": 8e-7,
    }
    assert inferred_series["tensorial_quadratic"]["inference_regularization"] == {
        "lambda_L": 0.0,
        "lambda_Q": 4e-7,
    }
    assert "lifting_regularization_gamma" not in inferred_series["fixed_rank_linear"]
    assert "lambda_Q" not in inferred_series["fixed_rank_linear"]["inference_regularization"]


def test_instantaneous_metric_preserves_exact_mass_convention():
    mass = np.diag([2.0, 3.0])
    steady = np.array([1.0, 2.0])
    fom = np.array([[2.0, 1.0], [2.0, 4.0]])
    rom = np.array([[1.0, 1.0], [2.0, 2.0]])
    expected = np.sqrt(np.array([2.0, 12.0]) / 14.0)
    np.testing.assert_allclose(
        instantaneous_error_history(fom, rom, mass, steady),
        expected,
    )


def test_approved_convergence_metric_and_pod_energy_are_defined():
    time = np.array([0.0, 5.0, 10.0])
    fom = np.ones((2, 3))
    rom = np.zeros((2, 3))
    np.testing.assert_allclose(
        publication_convergence_metric(fom, rom, np.eye(2), time),
        1.0,
    )
    curves = pod_energy_curves(np.array([3.0, 1.0]))
    np.testing.assert_allclose(curves.eigenvalues, [9.0, 1.0])
    np.testing.assert_allclose(curves.retained_energy_fraction, [0.9, 1.0])
    np.testing.assert_allclose(curves.unresolved_energy_fraction, [0.1, 0.0])


def test_no_publication_case_uses_zero_initial_flux(catalog):
    serialized = catalog.canonical_json()
    assert '"benchmark_variant":"legacy_sigmoid"' in serialized
    assert '"status":"intentional_repository_deviation"' in serialized
    assert '"type":"zero"' not in serialized
    assert '"kind":"zero"' not in serialized
