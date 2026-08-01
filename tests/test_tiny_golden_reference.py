import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import scipy as sp
from scipy import linalg
import scipy.sparse as sparse

import Nonlinear_Manifold_ROM as rom
from Nonlinear_Manifold_ROM import NonlinearManifoldReducedModel


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIRECTORY = Path(__file__).resolve().parent / "golden"
REFERENCE_PATH = GOLDEN_DIRECTORY / "tiny_1d_reference.npz"
MANIFEST_PATH = GOLDEN_DIRECTORY / "tiny_1d_manifest.json"
GENERATOR_PATH = (
    Path(__file__).resolve().parent
    / "reference_generators"
    / "generate_tiny_1d_reference.py"
)


@pytest.fixture(scope="module")
def golden_manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_reference():
    with np.load(REFERENCE_PATH, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def _update_length_prefixed(hasher, payload):
    hasher.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    hasher.update(payload)


def _canonical_content_checksum(arrays):
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[name]))
        _update_length_prefixed(hasher, name.encode("utf-8"))
        _update_length_prefixed(hasher, array.dtype.str.encode("ascii"))
        shape = json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        _update_length_prefixed(hasher, shape)
        _update_length_prefixed(hasher, array.tobytes(order="C"))
    return hasher.hexdigest()


def _tolerance(manifest, group):
    values = manifest["recommended_tolerances"][group]
    return {"rtol": float(values["rtol"]), "atol": float(values["atol"])}


def test_golden_artifact_integrity_and_authority(golden_manifest, golden_reference):
    assert golden_manifest["schema_version"] == "1.0.0"
    assert golden_manifest["artifact_name"] == REFERENCE_PATH.name
    assert golden_manifest["publication_reference"] is False
    assert golden_manifest["authority"] == {
        "artifact": "independent_numerical",
        "permitted_classifications": ["analytic", "independent_numerical"],
        "regression_only_content": False,
    }
    assert golden_manifest["generator"]["imports_production_modules"] is False
    assert golden_manifest["problem"]["production_sigmoid_represented"] is False
    assert set(golden_reference) == set(golden_manifest["arrays"])
    assert REFERENCE_PATH.stat().st_size < 1_000_000

    permitted = {"analytic", "independent_numerical"}
    for name, metadata in golden_manifest["arrays"].items():
        array = golden_reference[name]
        assert metadata["authority"] in permitted
        assert tuple(metadata["shape"]) == array.shape
        assert np.dtype(metadata["dtype"]) == array.dtype
        assert metadata["checksum_dtype_string"] == array.dtype.str
        assert np.all(np.isfinite(array))

    expected_checksum = golden_manifest["content_checksum"]["sha256"]
    assert golden_manifest["content_checksum"]["algorithm"] == (
        "sha256-canonical-array-content-v1"
    )
    assert _canonical_content_checksum(golden_reference) == expected_checksum


def test_generator_has_only_allowed_imports():
    tree = ast.parse(GENERATOR_PATH.read_text(encoding="utf-8"))
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots <= {
        "__future__",
        "argparse",
        "datetime",
        "hashlib",
        "json",
        "pathlib",
        "sys",
        "numpy",
        "scipy",
    }


def test_production_quadrature_and_mesh_match_analytic_reference(
    tiny_transport_operators, golden_manifest, golden_reference
):
    analytic_tolerance = _tolerance(golden_manifest, "analytic")
    np.testing.assert_allclose(
        tiny_transport_operators.quadrature.mu_q,
        golden_reference["mu"],
        **analytic_tolerance,
    )
    np.testing.assert_allclose(
        tiny_transport_operators.quadrature.w_q,
        golden_reference["weights"],
        **analytic_tolerance,
    )
    np.testing.assert_allclose(
        tiny_transport_operators.mesh.x,
        golden_reference["cell_edges"],
        **analytic_tolerance,
    )
    np.testing.assert_array_equal(
        tiny_transport_operators.mesh.cell2mat,
        golden_reference["cell_material_ids"],
    )
    assert np.sum(tiny_transport_operators.quadrature.w_q) == pytest.approx(
        1.0, abs=analytic_tolerance["atol"]
    )


def test_production_global_operators_match_independent_dense_assembly(
    tiny_transport_operators, golden_manifest, golden_reference
):
    operators = tiny_transport_operators
    production = {
        "mass_matrix": operators.mass.toarray(),
        "streaming_matrix": operators.streaming.toarray(),
        "total_interaction_matrix": operators.total.toarray(),
        "scattering_matrix": operators.scattering.toarray(),
        "system_matrix": operators.operator.toarray(),
        "boundary_inflow_matrix": np.asarray(operators.boundary),
    }
    boundary_values = np.zeros(operators.quadrature.ndir)
    boundary_values[-1] = 1.0
    production["boundary_source"] = operators.boundary @ boundary_values
    operator_tolerance = _tolerance(golden_manifest, "operators")

    for name, actual in production.items():
        expected = golden_reference[name]
        np.testing.assert_allclose(actual, expected, **operator_tolerance)
        np.testing.assert_array_equal(actual != 0.0, expected != 0.0)

    n_spatial = 12
    mass = production["mass_matrix"]
    streaming = production["streaming_matrix"]
    np.testing.assert_allclose(mass, mass.T, **operator_tolerance)
    assert np.min(np.linalg.eigvalsh(mass)) > 0.0
    for output_direction in range(4):
        rows = slice(output_direction * n_spatial, (output_direction + 1) * n_spatial)
        for input_direction in range(4):
            cols = slice(
                input_direction * n_spatial, (input_direction + 1) * n_spatial
            )
            if output_direction != input_direction:
                np.testing.assert_array_equal(streaming[rows, cols], 0.0)
    assert np.count_nonzero(production["boundary_source"]) == 1
    assert production["boundary_source"][3 * n_spatial] > 0.0


def test_production_steady_state_matches_independent_dense_solve(
    tiny_transport_operators, golden_manifest, golden_reference
):
    system = tiny_transport_operators.operator
    source = golden_reference["boundary_source"]
    production_steady = sparse.linalg.spsolve(system, source)
    reference_steady = golden_reference["steady_state"]
    steady_tolerance = _tolerance(golden_manifest, "steady_state")

    np.testing.assert_allclose(
        production_steady, reference_steady, **steady_tolerance
    )
    production_residual = np.linalg.norm(system @ production_steady - source) / np.linalg.norm(
        source
    )
    reference_residual = np.linalg.norm(
        golden_reference["system_matrix"] @ reference_steady - source
    ) / np.linalg.norm(source)
    assert production_residual < 1.0e-12
    assert reference_residual < 1.0e-12
    assert np.all(np.isfinite(production_steady))
    assert np.all(np.isfinite(reference_steady))


def test_production_radau_transient_matches_independent_matrix_exponential(
    tiny_transport_operators, golden_manifest, golden_reference
):
    mass = tiny_transport_operators.mass
    system = tiny_transport_operators.operator
    source = golden_reference["boundary_source"]
    time = golden_reference["time"]
    rhs = lambda current_time, state: sparse.linalg.spsolve(
        mass, source - system @ state
    )
    result = sp.integrate.solve_ivp(
        rhs,
        (time[0], time[-1]),
        np.zeros(mass.shape[0]),
        method="Radau",
        t_eval=time,
        atol=1.0e-10,
        rtol=1.0e-8,
    )

    assert result.success, result.message
    np.testing.assert_array_equal(result.t, time)
    assert result.y.shape == (48, 6)
    assert np.all(np.isfinite(result.y))
    np.testing.assert_allclose(
        result.y,
        golden_reference["transient_state"],
        **_tolerance(golden_manifest, "transient_state"),
    )


def test_production_pod_matches_independent_correlation_reference(
    monkeypatch, golden_manifest, golden_reference
):
    mass = golden_reference["mass_matrix"]
    mass_sqrt = np.real_if_close(linalg.sqrtm(mass))
    centered = golden_reference["transient_state"] - golden_reference[
        "steady_state"
    ][:, None]
    monkeypatch.setattr(rom, "globalMM", sparse.csc_matrix(mass))
    monkeypatch.setattr(rom, "globalMMsqrt", sparse.csc_matrix(mass_sqrt))

    model = NonlinearManifoldReducedModel(None)
    model.global_training_set = centered
    model.compute_pod(size_R=3, size_Q=0)

    eigenvalues = model.svd_val**2
    retained_energy = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    unresolved_energy = np.maximum(0.0, 1.0 - retained_energy)
    projector = model.pod_linear_basis @ model.pod_linear_basis.T @ mass
    residual = centered - projector @ centered
    projection_error = np.sqrt(
        np.maximum(0.0, np.einsum("ij,ij->j", residual, mass @ residual))
    )

    np.testing.assert_allclose(
        eigenvalues,
        golden_reference["pod_eigenvalues"],
        **_tolerance(golden_manifest, "pod_spectrum"),
    )
    np.testing.assert_allclose(
        retained_energy,
        golden_reference["pod_retained_energy"],
        **_tolerance(golden_manifest, "pod_spectrum"),
    )
    np.testing.assert_allclose(
        unresolved_energy,
        golden_reference["pod_unresolved_energy"],
        **_tolerance(golden_manifest, "pod_spectrum"),
    )
    np.testing.assert_allclose(
        projector,
        golden_reference["pod_projector_rank3"],
        **_tolerance(golden_manifest, "pod_projector"),
    )
    np.testing.assert_allclose(
        projection_error,
        golden_reference["pod_projection_error_rank3"],
        **_tolerance(golden_manifest, "pod_projection_error"),
    )


def test_reference_generator_check_mode_is_read_only():
    artifact_before = REFERENCE_PATH.read_bytes()
    manifest_before = MANIFEST_PATH.read_bytes()
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--check"],
        cwd=REPOSITORY_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "reference check passed: 19 arrays" in result.stdout
    assert REFERENCE_PATH.read_bytes() == artifact_before
    assert MANIFEST_PATH.read_bytes() == manifest_before
