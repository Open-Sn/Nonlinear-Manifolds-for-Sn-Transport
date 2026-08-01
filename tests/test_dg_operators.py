from types import SimpleNamespace

import numpy as np

from FLXSLV import FLXSLV
from MESH import MESH


def _one_direction_solver(mu):
    quadrature = SimpleNamespace(
        ndir=1, mu_q=np.array([mu]), w_q=np.array([1.0])
    )
    mesh = MESH(np.array([2.0]), np.array([2], dtype=int))
    return FLXSLV(
        quadrature,
        mesh,
        {"sigt": np.array([0.0]), "sigs": np.array([0.0])},
        {"qext": np.array([0.0]), "psi_bc": np.array([0.0])},
    )


def test_linear_dg_mass_matrix_independent_reference():
    solver = _one_direction_solver(0.5)
    mass = solver.compute_mass_matrix().toarray()

    # For h=1 and linear endpoint basis functions,
    # integral(phi_i phi_j dx) = [[1/3, 1/6], [1/6, 1/3]].
    expected_local = np.array([[1.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0 / 3.0]])
    np.testing.assert_allclose(mass[:2, :2], expected_local, atol=1e-14)
    np.testing.assert_allclose(mass[2:, 2:], expected_local, atol=1e-14)
    np.testing.assert_allclose(mass, mass.T, atol=0.0)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(expected_local), [1.0 / 6.0, 1.0 / 2.0], atol=1e-14
    )
    assert np.linalg.det(expected_local) > 0.0


def test_positive_direction_streaming_and_left_inflow():
    solver = _one_direction_solver(0.5)
    streaming, boundary = solver.assemble_global_grad_matrix(np.ones(1))

    expected = np.array(
        [
            [0.25, 0.25, 0.0, 0.0],
            [-0.25, 0.25, 0.0, 0.0],
            [0.0, -0.5, 0.25, 0.25],
            [0.0, 0.0, -0.25, 0.25],
        ]
    )
    expected_boundary = np.array([[0.5], [0.0], [0.0], [0.0]])
    np.testing.assert_allclose(streaming.toarray(), expected, atol=1e-14)
    np.testing.assert_allclose(boundary, expected_boundary, atol=1e-14)


def test_negative_direction_streaming_and_right_inflow():
    solver = _one_direction_solver(-0.5)
    streaming, boundary = solver.assemble_global_grad_matrix(np.ones(1))

    expected = np.array(
        [
            [0.25, -0.25, 0.0, 0.0],
            [0.25, 0.25, -0.5, 0.0],
            [0.0, 0.0, 0.25, -0.25],
            [0.0, 0.0, 0.25, 0.25],
        ]
    )
    expected_boundary = np.array([[0.0], [0.0], [0.0], [0.5]])
    np.testing.assert_allclose(streaming.toarray(), expected, atol=1e-14)
    np.testing.assert_allclose(boundary, expected_boundary, atol=1e-14)


def test_tiny_global_operator_dimensions_and_blocks(tiny_transport_operators):
    operators = tiny_transport_operators
    n_spatial = 2 * operators.mesh.n_cells
    n_phase = operators.quadrature.ndir * n_spatial

    assert operators.mass.shape == (n_phase, n_phase)
    assert operators.streaming.shape == (n_phase, n_phase)
    assert operators.total.shape == (n_phase, n_phase)
    assert operators.scattering.shape == (n_phase, n_phase)
    assert operators.boundary.shape == (n_phase, operators.quadrature.ndir)

    mass = operators.mass.toarray()
    total = operators.total.toarray()
    scattering = operators.scattering.toarray()
    streaming = operators.streaming.toarray()

    # Direction-major reaction blocks: identical spatial operators on diagonal.
    for direction in range(4):
        block = slice(direction * n_spatial, (direction + 1) * n_spatial)
        np.testing.assert_allclose(
            mass[block, block], operators.spatial_mass.toarray(), atol=1e-14
        )
        np.testing.assert_allclose(
            total[block, block], operators.spatial_total.toarray(), atol=1e-14
        )
    np.testing.assert_allclose(total[:n_spatial, n_spatial:], 0.0, atol=0.0)

    # Isotropic scattering maps every input direction into every output
    # direction, weighted only by the input ordinate weight.
    for output_direction in range(4):
        rows = slice(output_direction * n_spatial, (output_direction + 1) * n_spatial)
        for input_direction in range(4):
            cols = slice(input_direction * n_spatial, (input_direction + 1) * n_spatial)
            expected = (
                operators.quadrature.w_q[input_direction]
                * operators.spatial_scattering.toarray()
            )
            np.testing.assert_allclose(scattering[rows, cols], expected, atol=1e-14)

    # Streaming has no cross-direction coupling, but does have nearest-cell
    # upwind coupling inside each direction block.
    np.testing.assert_allclose(streaming[:n_spatial, n_spatial:], 0.0, atol=0.0)
    positive_block = streaming[3 * n_spatial : 4 * n_spatial, 3 * n_spatial : 4 * n_spatial]
    assert positive_block[2, 1] < 0.0
