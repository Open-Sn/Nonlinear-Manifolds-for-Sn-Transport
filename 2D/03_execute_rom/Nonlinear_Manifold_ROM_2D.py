"""
Nonlinear_Manifold_ROM_2D.py
---------------------------------
Semi-intrusive nonlinear-manifold reduced models for the two-dimensional
transport example.

The organization follows ``Nonlinear_Manifold_ROM.py``: one mutable model is
built in stages and the numerical experiments are written explicitly in the
main program.  Paths and test switches are ordinary research parameters near
the bottom of this file; there is no command-line interface.

The four tests create

    average_approximation_error_16_2d.pdf
    relative_error_inferred_models_16_2d.pdf
    Projected_Integral_Errors_32_2d.pdf
    scalar_flux_and_spatial_model_errors_16_2d.pdf

Exact semi-intrusive reproduction uses the centered snapshots, the steady
state, the spatial mass/reaction matrices, and a VTU mesh for the last plot.
"""

import copy
import time
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as tri
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import vtk
from vtk.util.numpy_support import vtk_to_numpy


# Fixed discretization used by the two-dimensional paper experiment.
NUM_DIRECT = 32
TT = 10.0
DT = 0.01
NUM_MODES = 132


class NonlinearManifoldReducedModel:
    """Semi-intrusive nonlinear-manifold ROM for the 2-D snapshot set."""

    def __init__(self, nonlinear_embedding_type=None):
        self.nonlinear_embedding_type = nonlinear_embedding_type
        self.nonlinear_function = lambda matrix: None

    # ======================================================================
    # 1. LOAD TRAINING DATA
    # ======================================================================
    def load_training_data(
        self,
        snapshot_path,
        center_path,
        mass_path,
        total_path,
        scattering_path,
        fission_path,
    ):
        """Load snapshots, the steady state, and the spatial operators."""

        # Load the centered angular-flux snapshots and the steady state.
        self.global_training_set = np.load(snapshot_path)["X"]
        self.solutionInf = np.load(center_path)

        # Load the spatial mass and reaction matrices.
        self.directMM = sparse.load_npz(mass_path).tocsc()
        self.directAbsorption = (
            sparse.load_npz(total_path) - sparse.load_npz(fission_path)
        ).tocsc()
        self.directScattering = sparse.load_npz(scattering_path).tocsc()

        # Store the phase-space and time dimensions.
        self.num_direct = NUM_DIRECT
        self.num_points = self.directMM.shape[0]
        self.num_snapshots = self.global_training_set.shape[1]
        self.TT = TT
        self.dt = DT
        self.time_steps = np.linspace(
            0.0, self.TT - self.dt, self.num_snapshots
        )

        print(f"Training set shape: {self.global_training_set.shape}")

    # ======================================================================
    # 2. MASS-WEIGHTED POD
    # ======================================================================
    def compute_pod(self):
        """Compute the special-first-vector POD used by the 2-D experiments."""

        # Compute snapshot energies, first-snapshot products, and the
        # normalization energies without assembling a global Kronecker mass.
        snapshot_energy = np.zeros(self.num_snapshots)
        first_products = np.zeros(self.num_snapshots)
        center_products = np.zeros(self.num_snapshots)
        center_energy = 0.0

        for dd in range(self.num_direct):
            rows = slice(dd * self.num_points, (dd + 1) * self.num_points)
            snapshots = self.global_training_set[rows, :]
            center = self.solutionInf[rows]
            weighted_snapshots = self.directMM @ snapshots
            weighted_center = self.directMM @ center

            snapshot_energy += np.sum(
                snapshots * weighted_snapshots, axis=0
            )
            first_products += snapshots[:, 0] @ weighted_snapshots
            center_products += center @ weighted_snapshots
            center_energy += center @ weighted_center

        # Normalize the first snapshot and remove it from the training set.
        first_norm = np.sqrt(first_products[0])
        first_coefficients = first_products / first_norm

        # Build the block mass square root W satisfying W.T @ W = M.
        mass_blocks = []
        for ii in range(0, self.num_points, 3):
            local_mass = self.directMM[ii : ii + 3, ii : ii + 3].toarray()
            mass_blocks.append(np.linalg.cholesky(local_mass).T)
        directMMsqrt = sparse.block_diag(mass_blocks, format="csr")

        # Reduce the weighted snapshot matrix direction by direction.  This is
        # the stable TSQR form of the same mass-weighted SVD used in the 1-D
        # driver and avoids materializing a second five-gigabyte snapshot set.
        reduced_factors = []
        for dd in range(self.num_direct):
            rows = slice(dd * self.num_points, (dd + 1) * self.num_points)
            weighted = directMMsqrt @ self.global_training_set[rows, :]
            weighted -= np.outer(
                weighted[:, 0] / first_norm, first_coefficients
            )
            reduced_factors.append(np.linalg.qr(weighted, mode="r"))
            print(f"  POD direction {dd + 1:2d}/{self.num_direct}")

        reduced_matrix = np.linalg.qr(
            np.vstack(reduced_factors), mode="r"
        )
        _, self.svd_val, temporal_transpose = np.linalg.svd(
            reduced_matrix, full_matrices=False
        )
        temporal_vectors = temporal_transpose.T

        # Materialize only the POD modes needed by the paper experiments.
        self.num_modes = NUM_MODES
        self.basis = np.empty(
            (self.global_training_set.shape[0], self.num_modes)
        )
        self.basis[:, 0] = self.global_training_set[:, 0] / first_norm
        for dd in range(self.num_direct):
            rows = slice(dd * self.num_points, (dd + 1) * self.num_points)
            modified = self.global_training_set[rows, :] - np.outer(
                self.basis[rows, 0], first_coefficients
            )
            self.basis[rows, 1:] = (
                modified @ temporal_vectors[:, : self.num_modes - 1]
            ) / self.svd_val[: self.num_modes - 1]

        # POD coefficients follow directly from the weighted SVD.
        self.coefficients = np.vstack(
            (
                first_coefficients,
                self.svd_val[: self.num_modes - 1, None]
                * temporal_vectors[:, : self.num_modes - 1].T,
            )
        )

        # Store the unresolved snapshot energy and the normalizations used by
        # Figures 9 and 10.
        represented_energy = np.sum(self.coefficients**2, axis=0)
        self.unresolved_energy = np.maximum(
            snapshot_energy - represented_energy, 0.0
        )
        full_solution_energy = (
            snapshot_energy + 2.0 * center_products + center_energy
        )
        self.refInfNorm = np.sqrt(center_energy)
        self.refTotNorm = np.sqrt(np.sum(full_solution_energy))
        self.snapshot_energy = snapshot_energy

        # Project the cell-local reaction operator onto all retained POD modes.
        basis_blocks = self.basis.reshape(
            self.num_direct, self.num_points, self.num_modes
        )
        projected_absorption = np.zeros((self.num_modes, self.num_modes))
        for dd in range(self.num_direct):
            projected_absorption += (
                basis_blocks[dd].T
                @ self.directAbsorption
                @ basis_blocks[dd]
            )
        summed_basis = np.sum(basis_blocks, axis=0)
        projected_scattering = (
            summed_basis.T @ self.directScattering @ summed_basis
        ) / self.num_direct
        self.projectedReaction = (
            projected_absorption - projected_scattering
        )

        print(f"POD done. Retained {self.num_modes} modes.")

    # ======================================================================
    # 3. NUMERICALLY APPROXIMATE TIME DERIVATIVES
    # ======================================================================
    def compute_time_derivatives(self):
        """Compute eighth-order finite-difference coefficient derivatives."""

        # Define the 9-point finite-difference stencils.
        central = np.array(
            [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0, 4 / 5,
             -1 / 5, 4 / 105, -1 / 280]
        ) / self.dt
        forward = np.array(
            [-761 / 280, 8, -14, 56 / 3, -35 / 2, 56 / 5,
             -14 / 3, 8 / 7, -1 / 8]
        ) / self.dt
        backward = np.array(
            [1 / 8, -8 / 7, 14 / 3, -56 / 5, 35 / 2, -56 / 3,
             14, -8, 761 / 280]
        ) / self.dt

        # Differentiation commutes with the fixed POD projection, so only the
        # small coefficient matrix is differentiated.
        NN = self.num_snapshots
        self.coefficient_derivatives = np.zeros_like(self.coefficients)
        for ii in range(4):
            self.coefficient_derivatives[:, ii] = (
                self.coefficients[:, ii : ii + 9] @ forward
            )
            start = NN - 12 + ii
            self.coefficient_derivatives[:, NN - 4 + ii] = (
                self.coefficients[:, start : start + 9] @ backward
            )
        for ii in range(NN - 8):
            self.coefficient_derivatives[:, ii + 4] = (
                self.coefficients[:, ii : ii + 9] @ central
            )

        print("Time derivatives computed (8th-order FD).")

    def select_pod_subspaces(self, size_R, size_Q):
        """Select reduced subspaces from the shared POD decomposition."""

        self.size_R = size_R
        self.size_Q = size_Q
        R, Q = self.size_R, self.size_Q

        self.pod_linear_coeff = self.coefficients[:R, :]
        self.pod_ortho_coeff = self.coefficients[R : R + Q, :]
        self.pod_global_coeff = self.coefficients[: R + Q, :]

    # ======================================================================
    # 4. COMPUTE NONLINEAR EMBEDDING
    # ======================================================================
    def compute_nonlinear_embedding(self, lambda_E):
        """Fit the polynomial or tensorial lifting into the POD complement."""

        if self.nonlinear_embedding_type == "poly":
            self.nonlinear_function = lambda matrix: matrix**2

        if self.nonlinear_embedding_type == "tens":
            upper = np.triu_indices(self.size_R)
            self.nonlinear_function = lambda matrix: np.einsum(
                "i...,j...->ij...", matrix, matrix
            )[upper[0], upper[1], ...]

        # Fit q = E h(a) with the normalized regularization used in the paper.
        self.pod_nonlinear_coeff = self.nonlinear_function(
            self.pod_linear_coeff
        )
        feature_gram = (
            self.pod_nonlinear_coeff @ self.pod_nonlinear_coeff.T
            + lambda_E
            * self.num_snapshots
            * np.eye(self.pod_nonlinear_coeff.shape[0])
        )
        self.nonlinear_lift_matrix = (
            np.linalg.solve(feature_gram, self.pod_nonlinear_coeff)
            @ self.pod_ortho_coeff.T
        ).T

    # ======================================================================
    # 5. PROJECT REACTION OPERATORS
    # ======================================================================
    def compute_projected_operators(self):
        """Select the projected derivative and reaction operators."""

        R, Q = self.size_R, self.size_Q
        self.projectedDerivativeLinear = self.coefficient_derivatives[:R, :]
        self.projectedReactionLinear = self.projectedReaction[:R, :R]

        if self.nonlinear_embedding_type is not None:
            self.projectedReactionNonlinear = (
                self.projectedReaction[:R, R : R + Q]
                @ self.nonlinear_lift_matrix
            )

    # ======================================================================
    # 6. INFER STREAMING OPERATORS
    # ======================================================================
    @staticmethod
    def linear_inference(linear_coeff, residual, lambda_A=0.0):
        """Infer the reduced linear streaming operator."""

        right_inverse = np.linalg.solve(
            linear_coeff @ linear_coeff.T
            + lambda_A * np.eye(linear_coeff.shape[0]),
            linear_coeff,
        ).T
        return -residual @ right_inverse

    @staticmethod
    def nonlinear_inference(
        linear_coeff,
        nonlinear_coeff,
        residual,
        lambda_A=0.0,
        lambda_H=1.0e-5,
        tolerance=1.0e-6,
    ):
        """Infer coupled linear/nonlinear streaming operators by iteration."""

        # Form the two regularized right inverses.
        right_linear = np.linalg.solve(
            linear_coeff @ linear_coeff.T
            + lambda_A * np.eye(linear_coeff.shape[0]),
            linear_coeff,
        ).T
        right_nonlinear = np.linalg.solve(
            nonlinear_coeff @ nonlinear_coeff.T
            + lambda_H * np.eye(nonlinear_coeff.shape[0]),
            nonlinear_coeff,
        ).T

        # Precompute the four blocks in the alternating update.
        simple_linear = -residual @ right_linear
        simple_nonlinear = -residual @ right_nonlinear
        nonlinear_linear = -nonlinear_coeff @ right_linear
        linear_nonlinear = -linear_coeff @ right_nonlinear

        inferred_linear = np.zeros(
            (linear_coeff.shape[0], linear_coeff.shape[0])
        )
        inferred_nonlinear = np.zeros(
            (linear_coeff.shape[0], nonlinear_coeff.shape[0])
        )

        count = 0
        while True:
            count += 1
            new_linear = (
                simple_linear
                + inferred_nonlinear @ nonlinear_linear
            )
            new_nonlinear = (
                simple_nonlinear + new_linear @ linear_nonlinear
            )

            error_linear = np.linalg.norm(
                new_linear - inferred_linear
            ) / (np.linalg.norm(new_linear) + np.finfo(float).eps)
            error_nonlinear = np.linalg.norm(
                new_nonlinear - inferred_nonlinear
            ) / (np.linalg.norm(new_nonlinear) + np.finfo(float).eps)

            inferred_linear = new_linear.copy()
            inferred_nonlinear = new_nonlinear.copy()
            if np.sqrt(error_linear**2 + error_nonlinear**2) < tolerance:
                break

        print(f"  Operator inference iterations: {count}")
        return inferred_linear, inferred_nonlinear

    def compute_inferred_operators(self, lambda_A=0.0, lambda_H=0.0):
        """Infer streaming and add the projected reaction contribution."""

        residual_linear = (
            self.projectedDerivativeLinear
            + self.projectedReactionLinear @ self.pod_linear_coeff
        )

        if self.nonlinear_embedding_type is None:
            inferred_streaming = self.linear_inference(
                self.pod_linear_coeff, residual_linear, lambda_A
            )
            self.inferredLinear = (
                inferred_streaming + self.projectedReactionLinear
            )
            return

        residual_nonlinear = (
            residual_linear
            + self.projectedReactionNonlinear
            @ self.pod_nonlinear_coeff
        )
        inferred_linear, inferred_nonlinear = self.nonlinear_inference(
            self.pod_linear_coeff,
            self.pod_nonlinear_coeff,
            residual_nonlinear,
            lambda_A=lambda_A,
            lambda_H=lambda_H * self.num_snapshots,
        )
        self.inferredLinear = (
            inferred_linear + self.projectedReactionLinear
        )
        self.inferredNonlinear = (
            inferred_nonlinear + self.projectedReactionNonlinear
        )

    # ======================================================================
    # 7. COMPUTE INITIAL CONDITIONS
    # ======================================================================
    def compute_initial_conditions(self):
        """Compute the linear or nonlinear-manifold initial condition."""

        self.initial_condition = self.pod_linear_coeff[:, 0].copy()

        if self.nonlinear_embedding_type is not None:
            error = lambda state: np.linalg.norm(
                self.pod_global_coeff[:, 0]
                - np.concatenate(
                    (
                        state,
                        self.nonlinear_lift_matrix
                        @ self.nonlinear_function(state),
                    )
                )
            )
            self.initial_condition = sp.optimize.minimize(
                error,
                self.initial_condition,
                method="Nelder-Mead",
                tol=1.0e-12,
                options={"maxiter": 1000},
            )["x"]

    # ======================================================================
    # 8. SOLVE REDUCED PROBLEM
    # ======================================================================
    def solve(self, repeats=1):
        """Integrate the inferred ROM and return POD coefficients and time."""

        if self.nonlinear_embedding_type is None:
            right_hand_side = lambda tt, state: -self.inferredLinear @ state
        else:
            right_hand_side = lambda tt, state: -(
                self.inferredLinear @ state
                + self.inferredNonlinear @ self.nonlinear_function(state)
            )

        start = time.perf_counter()
        for _ in range(repeats):
            reduced_solution = sp.integrate.solve_ivp(
                right_hand_side,
                (0.0, self.TT),
                self.initial_condition,
                method="Radau",
                atol=1.0e-12,
                rtol=1.0e-8,
                t_eval=self.time_steps,
            ).y
            if self.nonlinear_embedding_type is not None:
                closure_solution = (
                    self.nonlinear_lift_matrix
                    @ self.nonlinear_function(reduced_solution)
                )
        online_time = (time.perf_counter() - start) / repeats

        solution = np.zeros_like(self.coefficients)
        solution[: self.size_R, :] = reduced_solution
        if self.nonlinear_embedding_type is not None:
            solution[
                self.size_R : self.size_R + self.size_Q, :
            ] = closure_solution
        return solution, online_time

    # ======================================================================
    # 9. COMPUTE ERRORS
    # ======================================================================
    def coefficient_error(self, solution):
        """Squared phase-space error at every snapshot time."""

        return (
            np.sum((solution - self.coefficients) ** 2, axis=0)
            + self.unresolved_energy
        )

    def compute_errors(self, solution):
        """Instantaneous error normalized by the steady-state mass norm."""

        return np.sqrt(self.coefficient_error(solution)) / self.refInfNorm

    def compute_integrated_error(self, solution):
        """Relative space-time mass error used in Figure 10."""

        return np.sqrt(np.sum(self.coefficient_error(solution))) / self.refTotNorm

    def compute_projection_error(self, dimension):
        """Mass-orthogonal POD projection baseline."""

        projected = np.sum(self.coefficients[:dimension, :] ** 2, axis=0)
        error = np.maximum(self.snapshot_energy - projected, 0.0)
        return np.sqrt(np.sum(error)) / self.refTotNorm


def make_model(
    prototype,
    size_R,
    size_Q,
    embedding_type,
    lambda_E=0.0,
    lambda_A=0.0,
    lambda_H=0.0,
):
    """Select one ROM from the shared POD data and infer its operators."""

    model = copy.copy(prototype)
    model.nonlinear_embedding_type = embedding_type
    model.select_pod_subspaces(size_R, size_Q)
    if embedding_type is not None:
        model.compute_nonlinear_embedding(lambda_E)
    model.compute_projected_operators()
    model.compute_inferred_operators(lambda_A, lambda_H)
    model.compute_initial_conditions()
    return model


def plot_unresolved_energy(model, size_R, size_Q, output_path):
    """Plot the relative unresolved POD energy (Figure 8)."""

    energy = model.svd_val**2
    relative = np.sqrt(np.cumsum(energy[::-1])[::-1] / np.sum(energy))
    dimensions = np.arange(1, relative.size + 1)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.semilogy(dimensions + 1, relative, "k")
    ax.fill_between(
        dimensions[: size_R + 1],
        0.95 * relative[: size_R + 1],
        alpha=0.4,
    )
    ax.fill_between(
        dimensions[size_R : size_R + size_Q],
        0.95 * relative[size_R : size_R + size_Q],
        color="pink",
        alpha=0.7,
    )
    ax.set_xlim((0, 160))
    ax.set_ylim((1.0e-16, 1.0))
    ax.set_xlabel(r"$N_r + N_q$")
    ax.set_ylabel(r"$\rho_{\mathrm{miss}}(N_r + N_q)$")
    ax.set_title("Relative Unresolved Energy")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_inferred_errors(model, solutions, output_path):
    """Plot the three normalized inferred-model errors (Figure 9)."""

    display_time = np.linspace(0.0, 5.0, model.num_snapshots)
    fig, ax = plt.subplots(figsize=(7.8, 2.75))
    for label, solution in zip(
        ("Linear", "Polynomial", "Tensorial"), solutions
    ):
        ax.plot(display_time, model.compute_errors(solution), label=label)
    ax.set_yscale("log")
    ax.set_xlabel("t")
    ax.set_ylabel("Normalized Error")
    ax.grid(True)
    ax.legend(loc="lower center")
    fig.tight_layout()
    ax.set_xlim((0.0, 5.0))
    ax.set_ylim((3.6e-4, 3.6e-2))
    fig.savefig(output_path)
    plt.close(fig)


def plot_dimension_study(
    size_R_values,
    size_Q_values,
    errors_R,
    times_R,
    errors_Q,
    times_Q,
    output_path,
):
    """Plot ROM accuracy and online speed-up (Figure 10)."""

    reference_time = 3600.0
    fig, axes = plt.subplots(2, 2, figsize=(7.25, 6.5))

    labels_R = (
        ("-<", "C2", r"Tensorial    ($N_q = 80-N_r$)"),
        ("->", "C1", r"Polynomial ($N_q = 80-N_r$)"),
        ("-v", "C0", "Linear"),
    )
    for error, online, (marker, color, label) in zip(
        errors_R, times_R, labels_R
    ):
        axes[0, 0].semilogy(
            size_R_values, error, marker, color=color, label=label
        )
        axes[1, 0].semilogy(
            size_R_values, reference_time / online,
            marker, color=color, label=label,
        )
    axes[0, 0].set_ylabel("Relative Time-Average Error")
    axes[0, 0].legend(loc="lower left", labelspacing=0.2)
    axes[0, 0].grid(which="both")
    axes[1, 0].set_xlabel(r"$N_r$")
    axes[1, 0].set_ylabel("Online Speed-Up")
    axes[1, 0].legend(labelspacing=0.2)
    axes[1, 0].grid(which="both")

    labels_Q = (
        ("-<", "C2", r"Tensorial    ($N_r = 32$)"),
        ("->", "C1", r"Polynomial ($N_r = 32$)"),
        ("-v", "C0", r"Linear ($N_r = 32$)"),
        ("-^", "C3", r"Linear ($N_r = 32+N_q$)"),
    )
    for error, online, (marker, color, label) in zip(
        errors_Q[:4], times_Q, labels_Q
    ):
        axes[0, 1].loglog(
            size_Q_values, error, marker, color=color, label=label
        )
        axes[1, 1].loglog(
            size_Q_values, reference_time / online,
            marker, color=color, label=label,
        )
    axes[0, 1].loglog(
        size_Q_values, errors_Q[4], "-o", color="C4", label="Projection"
    )
    axes[0, 1].legend(loc="lower left")
    axes[0, 1].grid(which="both")
    axes[1, 1].set_xlabel(r"$N_q$")
    axes[1, 1].legend(loc="upper left")
    axes[1, 1].grid(which="both")

    # Match the limits and tick placement used in the paper figure.
    axes[0, 0].set_ylim((0.2e-3, 0.2e0))
    axes[0, 1].set_ylim((1.0e-6, 0.4e-1))
    axes[1, 0].set_ylim((3.0e3, 1.5e5))
    axes[1, 1].set_ylim((3.0e3, 1.5e5))
    axes[0, 0].set_xlim((2, 34))
    axes[1, 0].set_xlim((2, 34))
    axes[0, 0].set_xticks(np.arange(0, 37, 4))
    axes[1, 0].set_xticks(np.arange(0, 37, 4))
    axes[0, 0].tick_params(labelbottom=False)
    axes[0, 1].tick_params(labelbottom=False, labelleft=False, labelright=True)
    axes[1, 1].tick_params(labelleft=False, labelright=True)

    fig.suptitle("Inferred Streaming Operators", y=0.96)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_scalar_flux_errors(model, solutions, mesh_path, output_path):
    """Plot scalar flux and signed spatial ROM errors (Appendix figure)."""

    indices = np.array([200, 400, 600, 800, 1000])
    reference_angular = (
        model.global_training_set[:, indices] + model.solutionInf[:, None]
    )
    reference = reference_angular.reshape(
        model.num_direct, model.num_points, indices.size
    ).mean(axis=0)

    fields = [reference]
    for solution in solutions:
        reconstruction = model.basis @ solution[:, indices]
        angular_error = (
            reconstruction - model.global_training_set[:, indices]
        )
        fields.append(
            angular_error.reshape(
                model.num_direct, model.num_points, indices.size
            ).mean(axis=0)
        )
    fields = np.stack(fields, axis=2)

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(mesh_path))
    reader.Update()
    grid = reader.GetOutput()
    points = vtk_to_numpy(grid.GetPoints().GetData())
    cells = vtk_to_numpy(grid.GetCells().GetData()).reshape(-1, 4)[:, 1:]
    mesh = tri.Triangulation(points[:, 0], points[:, 1], cells)
    base_cmap = plt.get_cmap("RdBu_r")
    reference_cmap = colors.LinearSegmentedColormap.from_list(
        "reference", base_cmap(np.linspace(0.5, 1.0, 256))
    )
    cmaps = (reference_cmap, base_cmap, base_cmap, base_cmap)
    limits = ((0.0, 0.2), (-0.01, 0.01), (-0.01, 0.01),
              (-0.0025, 0.0025))
    titles = ("Reference", "Linear", "Polynomial", "Tensorial")

    fig = plt.figure(figsize=(8.0, 10.0))
    grid = fig.add_gridspec(
        6, 4, height_ratios=[1, 1, 1, 1, 1, 0.08]
    )
    last_images = []
    for row in range(5):
        for column in range(4):
            ax = fig.add_subplot(grid[row, column])
            image = ax.tripcolor(
                mesh,
                fields[:, row, column],
                shading="gouraud",
                cmap=cmaps[column],
                vmin=limits[column][0],
                vmax=limits[column][1],
            )
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[column], fontsize=13)
            if column == 0:
                ax.set_ylabel(rf"$t={row + 1}$", fontsize=11)
            if row == 4:
                last_images.append(image)

    for column, image in enumerate(last_images):
        color_axis = fig.add_subplot(grid[5, column])
        colorbar = fig.colorbar(image, cax=color_axis, orientation="horizontal")
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((0, 0))
        colorbar.ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =============================================================================
# PAPER TESTS / ENTRY POINT
# =============================================================================
if __name__ == "__main__":

    # -------------------------------------------------------------------------
    # Input paths.  Change only these entries when the files are elsewhere or
    # have different names.
    SNAPSHOT_PATH = "../run/preparation/snapshots/psi_matrix_centered_fp64.npz"
    CENTER_PATH = "../run/preparation/snapshots/center_col.npy"
    OPERATOR_PATH = Path("../run/preparation/full_order_out")
    MASS_PATH = OPERATOR_PATH / "ops_Mv.npz"
    TOTAL_PATH = OPERATOR_PATH / "ops_Mt.npz"
    SCATTERING_PATH = OPERATOR_PATH / "ops_Ms.npz"
    FISSION_PATH = OPERATOR_PATH / "ops_Mf.npz"
    MESH_PATH = "../run/opensn/aflux_3newss_1000_0.vtu"
    FIGURE_PATH = Path("../run/rom")
    FIGURE_PATH.mkdir(parents=True, exist_ok=True)

    # Select the paper experiments to run.
    TEST_1 = True    # relative unresolved energy
    TEST_2 = True    # inferred-model time histories
    TEST_3 = True    # dimension accuracy and online speed-up
    TEST_4 = True    # scalar-flux and spatial-error fields
    SIZE_R = 16
    SIZE_Q = 64

    # Build the training data, POD basis, and coefficient derivatives once.
    prototype_model = NonlinearManifoldReducedModel()
    prototype_model.load_training_data(
        SNAPSHOT_PATH,
        CENTER_PATH,
        MASS_PATH,
        TOTAL_PATH,
        SCATTERING_PATH,
        FISSION_PATH,
    )
    prototype_model.compute_pod()
    prototype_model.compute_time_derivatives()

    # =========================================================================
    # TEST 1 – RELATIVE UNRESOLVED ENERGY (FIGURE 8)
    # =========================================================================
    if TEST_1:
        print("\n--- TEST 1: Relative unresolved energy ---")
        plot_unresolved_energy(
            prototype_model,
            SIZE_R,
            SIZE_Q,
            FIGURE_PATH / "average_approximation_error_16_2d.pdf",
        )

    # Build the three N_r=16 models shared by Figures 9 and B.11.
    if TEST_2 or TEST_4:
        model_linear = make_model(
            prototype_model, SIZE_R, 0, None, lambda_A=0.0
        )
        model_polynomial = make_model(
            prototype_model,
            SIZE_R,
            SIZE_Q,
            "poly",
            lambda_E=1.2115282631363973e-7,
            lambda_A=0.0,
            lambda_H=4.6416e-6,
        )
        model_tensorial = make_model(
            prototype_model,
            SIZE_R,
            SIZE_Q,
            "tens",
            lambda_E=1.0e-8,
            lambda_A=0.0,
            lambda_H=2.1544e-7,
        )

        solution_linear, _ = model_linear.solve()
        solution_polynomial, _ = model_polynomial.solve()
        solution_tensorial, _ = model_tensorial.solve()
        main_solutions = (
            solution_linear,
            solution_polynomial,
            solution_tensorial,
        )

    # =========================================================================
    # TEST 2 – INFERRED MODEL ERRORS (FIGURE 9)
    # =========================================================================
    if TEST_2:
        print("\n--- TEST 2: Inferred model errors ---")
        plot_inferred_errors(
            prototype_model,
            main_solutions,
            FIGURE_PATH / "relative_error_inferred_models_16_2d.pdf",
        )

    # =========================================================================
    # TEST 3 – DIMENSION ACCURACY AND ONLINE SPEED-UP (FIGURE 10)
    # =========================================================================
    if TEST_3:
        print("\n--- TEST 3: Dimension accuracy and online speed-up ---")

        # Fixed-total-dimension study, N_q = 80 - N_r.
        NR_VALUES = np.array([4, 8, 12, 16, 20, 24, 28, 32])
        # Columns are gamma_tensor, gamma_poly, lambdaQ_tensor, lambdaQ_poly.
        PARAMETERS_NR = np.array([
            [2.1544e-7, 5.6234529783660859e-6,
             1.7782919726494412e-11, 5.6234529783660854e-5],
            [1.7782802973350419e-7, 1.2115081509900880e-6,
             9.9999696303823965e-6, 4.6416216240012703e-5],
            [1.2115282631363973e-7, 5.6233961738124829e-7,
             1.4678e-6, 1.4678e-5],
            [1.0e-8, 1.2115282631363973e-7, 2.1544e-7, 4.6416e-6],
            [1.7782919726494411e-9, 3.1623e-8,
             4.6415747372849845e-9, 2.1544e-6],
            [4.8232127617662724e-10, 1.9953e-8,
             5.2079702473670889e-9, 1.0391013288339083e-6],
            [2.5119e-10, 1.0e-8, 1.5849e-9, 7.9433e-7],
            [1.5849e-10, 1.5849e-9,
             6.3096e-10, 1.6469289708652770e-7],
        ])

        errors_R = [[], [], []]
        times_R = [[], [], []]
        for ii, size_R in enumerate(NR_VALUES):
            size_Q = 80 - size_R
            print(f"\n  N_r={size_R}, N_q={size_Q}")

            tensorial = make_model(
                prototype_model, size_R, size_Q, "tens",
                PARAMETERS_NR[ii, 0], 0.0, PARAMETERS_NR[ii, 2]
            )
            polynomial = make_model(
                prototype_model, size_R, size_Q, "poly",
                PARAMETERS_NR[ii, 1], 0.0, PARAMETERS_NR[ii, 3]
            )
            linear = make_model(
                prototype_model, size_R, 0, None, lambda_A=0.0
            )

            for jj, model in enumerate((tensorial, polynomial, linear)):
                solution, online_time = model.solve(repeats=50)
                errors_R[jj].append(model.compute_integrated_error(solution))
                times_R[jj].append(online_time)
                print(
                    f"    {('tensorial', 'polynomial', 'linear')[jj]:10s} "
                    f"error={errors_R[jj][-1]:.8e}, "
                    f"online={online_time:.5f}s"
                )

        # Fixed N_r=32 study.
        NQ_VALUES = np.array([1, 2, 4, 8, 16, 32, 64, 100])
        PARAMETERS_NQ = np.array([
            [1.5848999999999998e-9, 3.4145635402315319e-9,
             6.3096e-10, 1.646929e-7],
            [1.5849e-11, 3.4145635402315319e-9,
             6.3096e-10, 1.646929e-7],
            [1.5848999999999998e-9, 1.5849e-9,
             6.3096e-10, 1.646929e-7],
            [1.5849e-11, 1.5849e-9, 6.3096e-10, 1.646929e-7],
            [3.4145635402315318e-10, 1.5849e-9,
             6.3096e-10, 1.646929e-7],
            [1.5849e-10, 1.5849e-9, 6.3096e-10, 1.646929e-7],
            [1.5849e-10, 1.5849e-9, 6.3096e-10, 1.646929e-7],
            [1.5849e-10, 1.5849e-9, 6.3096e-10, 1.646929e-7],
        ])

        fixed_linear = make_model(
            prototype_model, 32, 0, None, lambda_A=1.0e-12
        )
        fixed_solution, fixed_time = fixed_linear.solve(repeats=50)
        fixed_error = fixed_linear.compute_integrated_error(fixed_solution)

        errors_Q = [[], [], [], [], []]
        times_Q = [[], [], [], []]
        for ii, size_Q in enumerate(NQ_VALUES):
            print(f"\n  N_r=32, N_q={size_Q}")

            tensorial = make_model(
                prototype_model, 32, size_Q, "tens",
                PARAMETERS_NQ[ii, 0], 0.0, PARAMETERS_NQ[ii, 2]
            )
            polynomial = make_model(
                prototype_model, 32, size_Q, "poly",
                PARAMETERS_NQ[ii, 1], 0.0, PARAMETERS_NQ[ii, 3]
            )
            expanded_linear = make_model(
                prototype_model, 32 + size_Q, 0, None,
                lambda_A=1.0e-12
            )

            for jj, model in enumerate((tensorial, polynomial)):
                solution, online_time = model.solve(repeats=50)
                errors_Q[jj].append(model.compute_integrated_error(solution))
                times_Q[jj].append(online_time)

            expanded_solution, expanded_time = expanded_linear.solve(repeats=50)
            errors_Q[2].append(fixed_error)
            errors_Q[3].append(
                expanded_linear.compute_integrated_error(expanded_solution)
            )
            errors_Q[4].append(
                prototype_model.compute_projection_error(32 + size_Q)
            )
            times_Q[2].append(fixed_time)
            times_Q[3].append(expanded_time)

            print(
                f"    tensorial error={errors_Q[0][-1]:.8e}\n"
                f"    polynomial error={errors_Q[1][-1]:.8e}\n"
                f"    expanded linear error={errors_Q[3][-1]:.8e}"
            )

        # Figure 10 plots the N_q=64 expanded-linear error again at N_q=100.
        errors_Q[3][-1] = errors_Q[3][-2]

        errors_R = tuple(np.asarray(values) for values in errors_R)
        times_R = tuple(np.asarray(values) for values in times_R)
        errors_Q = tuple(np.asarray(values) for values in errors_Q)
        times_Q = tuple(np.asarray(values) for values in times_Q)
        plot_dimension_study(
            NR_VALUES,
            NQ_VALUES,
            errors_R,
            times_R,
            errors_Q,
            times_Q,
            FIGURE_PATH / "Projected_Integral_Errors_32_2d.pdf",
        )

    # =========================================================================
    # TEST 4 – SCALAR FLUX AND SPATIAL MODEL ERRORS (FIGURE B.11)
    # =========================================================================
    if TEST_4:
        print("\n--- TEST 4: Scalar flux and spatial model errors ---")
        plot_scalar_flux_errors(
            prototype_model,
            main_solutions,
            MESH_PATH,
            FIGURE_PATH / "scalar_flux_and_spatial_model_errors_16_2d.pdf",
        )
