"""Configured construction and assembly stages for the existing 1-D problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import OneDConfig


@dataclass(frozen=True)
class ConfiguredProblem:
    """Geometry, quadrature, and material data before global assembly."""

    config: OneDConfig
    quadrature: Any
    mesh: Any
    cross_sections: dict[str, np.ndarray]
    sources: dict[str, np.ndarray]
    dof_coordinates: np.ndarray


@dataclass(frozen=True)
class AssembledOperators:
    """Sparse operators in direction-major phase-space ordering."""

    problem: ConfiguredProblem
    solver: Any
    spatial_mass: Any
    spatial_total_interaction: Any
    spatial_scattering: Any
    mass: Any
    inverse_mass: Any
    streaming: Any
    boundary_inflow_matrix: np.ndarray
    total_interaction: Any
    scattering: Any
    system: Any
    boundary_source: np.ndarray


def build_problem(config: OneDConfig) -> ConfiguredProblem:
    """Build configured mesh/quadrature objects without assembling global matrices."""
    from AQ import AQ
    from MESH import MESH

    problem_config = config.problem
    quadrature = AQ(problem_config.angular_ordinates)
    mesh = MESH(
        np.asarray(problem_config.region_widths, dtype=float),
        np.asarray(problem_config.cells_per_region, dtype=int),
    )
    cross_sections = {
        "ng": 1,
        "sigt": np.asarray(problem_config.sigma_t, dtype=float),
        "sigs": np.asarray(problem_config.sigma_s, dtype=float),
    }
    sources = {
        "ng": 1,
        "qext": np.zeros(len(problem_config.region_widths)),
        "psi_bc": np.zeros(problem_config.angular_ordinates),
    }
    # Preserve the endpoint-DG coordinate ordering used by the root driver.
    dof_coordinates = np.concatenate(
        [[mesh.x[index], mesh.x[index]] for index in range(len(mesh.x))]
    )[1:-1]
    return ConfiguredProblem(
        config=config,
        quadrature=quadrature,
        mesh=mesh,
        cross_sections=cross_sections,
        sources=sources,
        dof_coordinates=dof_coordinates,
    )


def construct_boundary_values(
    problem: ConfiguredProblem, current_time: float | None = None
) -> np.ndarray:
    """Construct the configured constant physical inflow vector."""
    del current_time  # reserved for future explicitly time-dependent configurations
    config = problem.config.problem
    values = np.zeros(config.angular_ordinates)
    half = config.angular_ordinates // 2
    if config.inflow_boundary == "left":
        if config.inflow_direction == "isotropic":
            indices = slice(half, None)
        elif config.inflow_direction == "most_grazing":
            indices = half
        else:
            indices = config.angular_ordinates - 1
    else:
        if config.inflow_direction == "isotropic":
            indices = slice(None, half)
        elif config.inflow_direction == "most_grazing":
            indices = half - 1
        else:
            indices = 0
    values[indices] = config.inflow_amplitude
    return values


def construct_initial_condition_values(
    config: OneDConfig, dof_coordinates: np.ndarray
) -> np.ndarray:
    """Construct configured initial values for an explicit spatial DOF array."""
    initial = config.problem.initial_condition
    n_directions = config.problem.angular_ordinates
    dof_coordinates = np.asarray(dof_coordinates)
    n_spatial = dof_coordinates.size
    if initial.kind == "zero":
        return np.zeros(n_directions * n_spatial)

    # This is the protected production formula expressed by explicit config.
    profile = initial.amplitude * (
        1.0
        - 1.0
        / (
            1.0
            + np.exp(
                -initial.steepness
                * (dof_coordinates - initial.transition_location)
            )
        )
    )
    angular_block = (
        n_directions - 1
        if initial.angular_block == "final"
        else int(initial.angular_block)
    )
    state = np.zeros((n_directions, n_spatial))
    state[angular_block, :] = profile
    return state.reshape(-1)


def construct_initial_condition(problem: ConfiguredProblem) -> np.ndarray:
    """Construct the configured state in direction-major ordering."""
    return construct_initial_condition_values(
        problem.config, problem.dof_coordinates
    )


def assemble_operators(problem: ConfiguredProblem) -> AssembledOperators:
    """Assemble the current sparse DG operators without changing their formulas."""
    import scipy.sparse as sparse

    from FLXSLV import FLXSLV

    config = problem.config.problem
    solver = FLXSLV(
        problem.quadrature,
        problem.mesh,
        problem.cross_sections,
        problem.sources,
    )
    spatial_mass = solver.assemble_global_mass_matrix(
        np.ones(len(config.region_widths))
    )
    spatial_total = solver.assemble_global_mass_matrix(
        np.asarray(config.sigma_t, dtype=float)
    )
    spatial_scattering = solver.assemble_global_mass_matrix(
        np.asarray(config.sigma_s, dtype=float)
    )
    identity = np.eye(config.angular_ordinates)
    mass = sparse.kron(identity, spatial_mass, format="csc")
    inverse_mass = sparse.kron(
        identity, sparse.linalg.inv(spatial_mass), format="csc"
    )
    total = sparse.kron(identity, spatial_total, format="csc")
    scattering = sparse.kron(
        np.tile(problem.quadrature.w_q, (config.angular_ordinates, 1)),
        spatial_scattering,
        format="csc",
    )
    streaming, boundary_map = solver.assemble_global_grad_matrix(
        np.ones(config.angular_ordinates)
    )
    system = streaming + total - scattering
    boundary_source = boundary_map @ construct_boundary_values(problem)
    return AssembledOperators(
        problem=problem,
        solver=solver,
        spatial_mass=spatial_mass,
        spatial_total_interaction=spatial_total,
        spatial_scattering=spatial_scattering,
        mass=mass,
        inverse_mass=inverse_mass,
        streaming=streaming,
        boundary_inflow_matrix=boundary_map,
        total_interaction=total,
        scattering=scattering,
        system=system,
        boundary_source=np.asarray(boundary_source),
    )


def mass_matrix_square_root(operators: AssembledOperators):
    """Build the same blockwise symmetric mass square root used by the ROM."""
    import copy
    import scipy as sp

    mass_sqrt = copy.deepcopy(operators.mass)
    block = lambda index: np.ix_([2 * index, 2 * index + 1], [2 * index, 2 * index + 1])
    for index in range(operators.mass.shape[0] // 2):
        indices = block(index)
        mass_sqrt[indices] = sp.linalg.fractional_matrix_power(
            operators.mass[indices].todense(), 0.5
        )
    return mass_sqrt


def mass_matrix_block_inverse(operators: AssembledOperators):
    """Build the historical phase-space matrix of inverted local mass blocks."""
    import copy

    mass_inverse = copy.deepcopy(operators.mass)
    block = lambda index: np.ix_([2 * index, 2 * index + 1], [2 * index, 2 * index + 1])
    for index in range(operators.mass.shape[0] // 2):
        indices = block(index)
        mass_inverse[indices] = np.linalg.inv(operators.mass[indices].todense())
    return mass_inverse
