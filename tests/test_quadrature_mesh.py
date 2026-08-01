import numpy as np

from AQ import AQ
from MESH import MESH


def test_aq4_matches_tabulated_gauss_legendre_rule():
    # Conventional GL4 weights integrate over [-1, 1] and sum to two. AQ
    # explicitly divides them by their sum, so stored transport weights are
    # one half of the conventional values and sum to one.
    expected_nodes = np.array(
        [
            -0.8611363115940526,
            -0.3399810435848563,
            0.3399810435848563,
            0.8611363115940526,
        ]
    )
    conventional_weights = np.array(
        [
            0.3478548451374539,
            0.6521451548625461,
            0.6521451548625461,
            0.3478548451374539,
        ]
    )
    quadrature = AQ(4)

    np.testing.assert_allclose(quadrature.mu_q, expected_nodes, atol=2e-15)
    np.testing.assert_allclose(
        quadrature.w_q, conventional_weights / 2.0, atol=2e-15
    )
    np.testing.assert_allclose(quadrature.mu_q, -quadrature.mu_q[::-1])
    np.testing.assert_allclose(quadrature.w_q, quadrature.w_q[::-1])
    assert np.all(quadrature.w_q > 0.0)
    assert np.isclose(np.sum(quadrature.w_q), 1.0, atol=2e-15)


def test_three_zone_mesh_geometry_and_material_mapping():
    mesh = MESH(
        np.array([1.0, 1.0, 1.0]), np.array([2, 2, 2], dtype=int)
    )

    np.testing.assert_array_equal(
        mesh.x, np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    )
    np.testing.assert_array_equal(mesh.dx, np.full(6, 0.5))
    np.testing.assert_array_equal(mesh.x[[0, 2, 4, 6]], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(mesh.cell2mat, [0, 0, 1, 1, 2, 2])
    assert mesh.n_zones == 3
    assert mesh.n_cells == 6
    assert 2 * mesh.n_cells == 12  # two linear-DG spatial DOFs per cell
