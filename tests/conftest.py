from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sparse

from AQ import AQ
from FLXSLV import FLXSLV
from MESH import MESH


@pytest.fixture
def tiny_transport_operators():
    """Six-cell paper-material operator set used only by fast tests."""
    quadrature = AQ(4)
    mesh = MESH(
        np.array([1.0, 1.0, 1.0]), np.array([2, 2, 2], dtype=int)
    )
    cross_sections = {
        "sigt": np.array([0.0, 1.0, 0.0]),
        "sigs": np.array([0.0, 0.99, 0.0]),
    }
    sources = {
        "qext": np.zeros(3),
        "psi_bc": np.zeros(quadrature.ndir),
    }
    solver = FLXSLV(quadrature, mesh, cross_sections, sources)
    spatial_mass = solver.assemble_global_mass_matrix(np.ones(3))
    spatial_total = solver.assemble_global_mass_matrix(cross_sections["sigt"])
    spatial_scattering = solver.assemble_global_mass_matrix(
        cross_sections["sigs"]
    )
    phase_mass = sparse.kron(
        sparse.eye(quadrature.ndir), spatial_mass, format="csc"
    )
    total = sparse.kron(
        sparse.eye(quadrature.ndir), spatial_total, format="csc"
    )
    scattering = sparse.kron(
        np.tile(quadrature.w_q, (quadrature.ndir, 1)),
        spatial_scattering,
        format="csc",
    )
    streaming, boundary = solver.assemble_global_grad_matrix(
        np.ones(quadrature.ndir)
    )
    return SimpleNamespace(
        quadrature=quadrature,
        mesh=mesh,
        solver=solver,
        spatial_mass=spatial_mass,
        spatial_total=spatial_total,
        spatial_scattering=spatial_scattering,
        mass=phase_mass,
        total=total,
        scattering=scattering,
        streaming=streaming,
        boundary=boundary,
        operator=streaming + total - scattering,
    )
