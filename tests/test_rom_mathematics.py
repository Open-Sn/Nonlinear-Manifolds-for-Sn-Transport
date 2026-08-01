import inspect

import numpy as np
import pytest
import scipy.sparse as sparse

import Nonlinear_Manifold_ROM as rom
from Nonlinear_Manifold_ROM import (
    DEFAULT_MAX_INFERENCE_ITERATIONS,
    NonlinearManifoldReducedModel,
    normalize_embedding_type,
    quadratic_features,
)
from one_d.nonlinear_diagnostics import (
    difference_metrics,
    historical_alternating_inference,
    historical_lifting,
    historical_nonlinear_rhs,
    historical_quadratic_features,
    latent_derivatives,
)


def test_production_inference_limit_exceeds_published_iteration_count():
    library_default = inspect.signature(
        NonlinearManifoldReducedModel.nonlinear_inference
    ).parameters["max_iterations"].default
    workflow_default = inspect.signature(
        NonlinearManifoldReducedModel.compute_inferred_operators
    ).parameters["max_iterations"].default

    assert DEFAULT_MAX_INFERENCE_ITERATIONS >= 100000
    assert library_default == DEFAULT_MAX_INFERENCE_ITERATIONS
    assert workflow_default == DEFAULT_MAX_INFERENCE_ITERATIONS
    assert workflow_default > 23367
    # The three production calls inherit the reviewed method default.
    assert "max_iterations=" not in inspect.getsource(rom.main)


def test_elementwise_quadratic_features_for_vector_and_snapshots():
    vector = np.array([1.0, 2.0, 3.0])
    snapshots = np.column_stack((vector, 2.0 * vector))

    np.testing.assert_array_equal(
        quadratic_features(vector, "elementwise"), [1.0, 4.0, 9.0]
    )
    np.testing.assert_array_equal(
        quadratic_features(snapshots, "elementwise"), snapshots**2
    )


def test_tensorial_quadratic_features_for_vector_and_snapshots():
    vector = np.array([1.0, 2.0, 3.0])
    expected = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0])
    snapshots = np.column_stack((vector, 2.0 * vector))

    vector_features = quadratic_features(vector, "tensorial")
    snapshot_features = quadratic_features(snapshots, "tensorial")
    np.testing.assert_array_equal(vector_features, expected)
    np.testing.assert_array_equal(snapshot_features[:, 0], expected)
    np.testing.assert_array_equal(snapshot_features[:, 1], 4.0 * expected)
    assert vector_features.shape == (6,)
    assert snapshot_features.shape == (6, 2)
    assert len(vector_features) == 3 * (3 + 1) // 2


@pytest.mark.parametrize("model", ["elementwise", "tensorial"])
def test_historical_feature_and_lifting_reference_matches_current(model):
    generator = np.random.default_rng(402)
    latent = generator.normal(size=(3, 20))
    lifting = generator.normal(size=(5, 20))
    ridge = 2.5e-4
    reference = historical_lifting(latent, lifting, ridge, model)

    current = NonlinearManifoldReducedModel(model)
    current.size_R = latent.shape[0]
    current.pod_linear_coeff = latent
    current.pod_ortho_coeff = lifting
    current.pod_ortho_basis = np.eye(lifting.shape[0])
    current.compute_nonlinear_embedding(lambda_E=ridge)

    np.testing.assert_array_equal(
        historical_quadratic_features(latent, model),
        quadratic_features(latent, model),
    )
    np.testing.assert_allclose(
        current.nonlinear_lift_matrix,
        reference["lifting_matrix"],
        rtol=2e-14,
        atol=2e-14,
    )


def test_historical_rhs_and_inference_sequence_match_current():
    linear, nonlinear, residual, _, _ = _known_nonlinear_inference_problem()
    reference = historical_alternating_inference(
        linear,
        nonlinear,
        residual,
        ridge_linear=0.0,
        ridge_nonlinear=0.0,
        tolerance=1e-11,
        maximum_iterations=10000,
        checkpoints=(1, 10, 100),
    )
    current_linear, current_nonlinear, current_diagnostics = (
        NonlinearManifoldReducedModel.nonlinear_inference(
            linear,
            nonlinear,
            residual,
            ll_A=0.0,
            ll_H=0.0,
            tolerance=1e-11,
            max_iterations=10000,
            return_diagnostics=True,
        )
    )

    np.testing.assert_allclose(reference["linear_operator"], current_linear, atol=1e-14)
    np.testing.assert_allclose(
        reference["nonlinear_operator"], current_nonlinear, atol=1e-14
    )
    assert reference["iteration_count"] == current_diagnostics["iteration_count"]
    assert reference["final_convergence_measure"] == pytest.approx(
        current_diagnostics["final_convergence_measure"], rel=2e-14
    )
    coordinate = np.array([0.2, -0.3])
    np.testing.assert_allclose(
        historical_nonlinear_rhs(
            coordinate, current_linear, current_nonlinear, "tensorial"
        ),
        -current_linear @ coordinate
        - current_nonlinear @ quadratic_features(coordinate, "tensorial"),
    )


def test_historical_derivative_defect_is_isolated_to_final_four_columns():
    time = np.arange(20, dtype=float) * 0.1
    values = np.vstack((time**2, time**3))
    corrected = latent_derivatives(values, 0.1)
    defective = latent_derivatives(values, 0.1, historical_defect=True)

    np.testing.assert_allclose(corrected[:, :-4], defective[:, :-4])
    assert np.linalg.norm(corrected[:, -4:] - defective[:, -4:]) > 1.0
    assert np.linalg.norm(defective[:, -1] - defective[:, -2]) == pytest.approx(0.0)
    assert difference_metrics(corrected, corrected) == {
        "maximum_absolute_difference": 0.0,
        "relative_frobenius_difference": 0.0,
    }


@pytest.mark.parametrize(
    ("name", "canonical"),
    [
        ("elementwise", "elementwise"),
        ("poly", "elementwise"),
        ("tensorial", "tensorial"),
        ("tens", "tensorial"),
        (None, None),
    ],
)
def test_embedding_names_are_normalized(name, canonical):
    assert normalize_embedding_type(name) == canonical
    assert NonlinearManifoldReducedModel(name).nonlinear_embedding_type == canonical


def test_invalid_embedding_name_lists_accepted_values():
    with pytest.raises(ValueError, match="elementwise.*tensorial.*poly.*tens"):
        NonlinearManifoldReducedModel("not-an-embedding")


def test_mass_weighted_pod_orthonormality_and_spectrum(monkeypatch):
    mass = np.diag([1.0, 2.0, 3.0, 4.0])
    mass_sqrt = np.diag(np.sqrt(np.diag(mass)))
    snapshots = np.array(
        [
            [1.0, 0.5, 0.0, -0.5, -1.0],
            [0.0, 1.0, 0.5, 0.0, -0.5],
            [1.0, 1.0, 0.0, 1.0, 1.0],
            [0.5, -0.5, 1.0, -1.0, 0.25],
        ]
    )
    monkeypatch.setattr(rom, "globalMM", sparse.csc_matrix(mass))
    monkeypatch.setattr(rom, "globalMMsqrt", sparse.csc_matrix(mass_sqrt))

    model = NonlinearManifoldReducedModel(None)
    model.global_training_set = snapshots
    model.compute_pod(size_R=3, size_Q=1)

    np.testing.assert_allclose(
        model.basis.T @ mass @ model.basis, np.eye(4), atol=2e-12
    )
    expected_singular_values = np.linalg.svd(
        mass_sqrt @ snapshots, compute_uv=False
    )
    np.testing.assert_allclose(model.svd_val, expected_singular_values, rtol=2e-13)

    projector = model.pod_linear_basis @ model.pod_linear_basis.T @ mass
    projected = projector @ snapshots
    assert np.linalg.norm(snapshots - projected) < np.linalg.norm(snapshots)


def test_synthetic_elementwise_lifting_recovers_known_map():
    coefficients = np.array(
        [[1.0, 2.0, 1.0, 3.0, 2.0, 4.0], [1.0, 1.0, 2.0, 1.0, 3.0, 2.0]]
    )
    features = coefficients**2
    expected_lift = np.array([[0.75, -0.25], [0.5, 1.25]])

    model = NonlinearManifoldReducedModel("elementwise")
    model.size_R = 2
    model.pod_linear_coeff = coefficients
    model.pod_ortho_coeff = expected_lift @ features
    model.pod_ortho_basis = np.eye(2)
    model.compute_nonlinear_embedding(lambda_E=0.0)

    np.testing.assert_allclose(
        model.nonlinear_lift_matrix, expected_lift, rtol=2e-13, atol=2e-13
    )


def test_linear_operator_inference_recovers_known_operator():
    coefficients = np.array(
        [[1.0, 0.0, 2.0, -1.0, 0.5], [0.0, 1.0, -1.0, 2.0, 1.5]]
    )
    expected_operator = np.array([[0.5, -0.2], [0.1, 0.7]])
    residual = -expected_operator @ coefficients

    inferred = NonlinearManifoldReducedModel.linear_inference(
        coefficients, residual, ll_A=0.0
    )
    np.testing.assert_allclose(inferred, expected_operator, atol=2e-14)
    np.testing.assert_allclose(residual + inferred @ coefficients, 0.0, atol=2e-14)


def _known_nonlinear_inference_problem():
    generator = np.random.default_rng(12345)
    linear = generator.uniform(-1.0, 1.0, size=(2, 200))
    nonlinear = quadratic_features(linear, "tensorial")
    expected_linear = np.array([[0.5, -0.2], [0.1, 0.7]])
    expected_nonlinear = np.array(
        [[0.2, -0.1, 0.05], [-0.3, 0.15, 0.2]]
    )
    residual = -expected_linear @ linear - expected_nonlinear @ nonlinear
    return linear, nonlinear, residual, expected_linear, expected_nonlinear


def test_nonlinear_operator_inference_converges_with_diagnostics():
    linear, nonlinear, residual, expected_linear, expected_nonlinear = (
        _known_nonlinear_inference_problem()
    )
    inferred_linear, inferred_nonlinear, diagnostics = (
        NonlinearManifoldReducedModel.nonlinear_inference(
            linear,
            nonlinear,
            residual,
            ll_A=0.0,
            ll_H=0.0,
            tolerance=1e-11,
            max_iterations=10000,
            return_diagnostics=True,
        )
    )

    assert diagnostics["converged"] is True
    assert diagnostics["termination_reason"] == "converged"
    assert 1 <= diagnostics["iteration_count"] < 10000
    assert diagnostics["final_convergence_measure"] < 1e-11
    np.testing.assert_allclose(inferred_linear, expected_linear, atol=5e-12)
    np.testing.assert_allclose(inferred_nonlinear, expected_nonlinear, atol=5e-12)
    np.testing.assert_allclose(
        residual + inferred_linear @ linear + inferred_nonlinear @ nonlinear,
        0.0,
        atol=1e-11,
    )


def test_nonlinear_operator_inference_stops_at_maximum_iterations():
    linear, nonlinear, residual, _, _ = _known_nonlinear_inference_problem()
    _, _, diagnostics = NonlinearManifoldReducedModel.nonlinear_inference(
        linear,
        nonlinear,
        residual,
        ll_A=0.0,
        ll_H=0.0,
        tolerance=0.0,
        max_iterations=1,
        return_diagnostics=True,
    )
    assert diagnostics["converged"] is False
    assert diagnostics["iteration_count"] == 1
    assert diagnostics["termination_reason"] == "maximum_iterations"
    assert np.isfinite(diagnostics["final_convergence_measure"])


def test_nonlinear_operator_inference_does_not_silently_return_nonconvergence():
    linear, nonlinear, residual, _, _ = _known_nonlinear_inference_problem()
    with pytest.raises(RuntimeError, match="maximum_iterations"):
        NonlinearManifoldReducedModel.nonlinear_inference(
            linear,
            nonlinear,
            residual,
            ll_A=0.0,
            ll_H=0.0,
            tolerance=0.0,
            max_iterations=1,
        )


def test_nonlinear_operator_inference_detects_nonfinite_iterate():
    linear, nonlinear, residual, _, _ = _known_nonlinear_inference_problem()
    residual = residual.copy()
    residual[0, :] = np.inf
    with np.errstate(over="ignore", invalid="ignore"):
        _, _, diagnostics = NonlinearManifoldReducedModel.nonlinear_inference(
            linear,
            nonlinear,
            residual,
            max_iterations=5,
            return_diagnostics=True,
        )
    assert diagnostics["converged"] is False
    assert diagnostics["termination_reason"] == "nonfinite_iterate"
    assert diagnostics["iteration_count"] == 1


def test_nonlinear_operator_inference_detects_nonfinite_measure(monkeypatch):
    linear, nonlinear, residual, _, _ = _known_nonlinear_inference_problem()
    monkeypatch.setattr(rom.np.linalg, "norm", lambda *args, **kwargs: np.nan)
    _, _, diagnostics = NonlinearManifoldReducedModel.nonlinear_inference(
        linear,
        nonlinear,
        residual,
        max_iterations=5,
        return_diagnostics=True,
    )
    assert diagnostics["converged"] is False
    assert diagnostics["termination_reason"] == "nonfinite_convergence_measure"
    assert diagnostics["iteration_count"] == 1


def test_current_error_definition_is_steady_state_normalized(monkeypatch):
    # The current convention is I_angle kron M_x; no angular weights are added.
    mass = sparse.diags([1.0, 2.0])
    monkeypatch.setattr(rom, "globalMM", mass)
    model = NonlinearManifoldReducedModel(None)
    model.solutionInf = np.array([2.0, 1.0])
    model.solutionDG1 = np.array([[1.0, 2.0], [1.0, 1.0]])
    reconstruction = np.array([[2.0, 2.0], [1.0, 2.0]])

    errors = model.compute_errors(reconstruction)
    steady_square_norm = 2.0**2 + 2.0 * 1.0**2
    expected = np.sqrt(np.array([1.0, 2.0]) / steady_square_norm)
    np.testing.assert_allclose(errors, expected, atol=1e-14)
