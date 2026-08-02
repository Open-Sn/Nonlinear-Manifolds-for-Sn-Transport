"""
NonlinearManifoldReducedModel.py
-----------
Nonlinear Manifold Reduced Model (NonlinearManifoldReducedModel): 
intrusive and semi-intrusive reduced-order modelling of a Sp-DG1 neutron transport solver.
 
Public methods of the class include:
    0.  __init__()                    - constructor method to initialize the class
    1.  load_training_data()          - load solution snapshots and build training set
    2.  compute_time_derivatives()    - use 8th-order finite-difference time derivatives
    3.  compute_pod()                 - extract POD / SVD bases from the training set
    4.  compute_nonlinear_embedding() - compute polynomial, tensorial, and RBF embeddings
    5.  compute_projected_operators() - project full-order operators onto reduced bases
    6.  compute_inferred_operators()  - use operator inference to approximate RB streaming operators
    7.  compute_initial_conditions()  - approximate states in the linear and nonlinear manifolds
    8.  solve()                       - time-integrate the projected/inferred reduced models
    9.  compute_errors()              - normalised (MM)-norm errors
 
Usage
-----
    from Nonlinear_Manifold_ROM import NonlinearManifoldReducedModel
    model = NonlinearManifoldReducedModel(nonlinear_embedding_type="tensorial")
"""

import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import copy as copy
import time as time
import os

import Transport_Driver_Benchmark_1D as transport_driver
from Transport_Driver_Benchmark_1D import *
from one_d.rom import partition_time_indices as _partition_time_indices


_EMBEDDING_ALIASES = {
    "elementwise": "elementwise",
    "poly": "elementwise",
    "tensorial": "tensorial",
    "tens": "tensorial",
    "rbf": "rbf",
}

# The published element-wise case required 23,367 alternating updates. Keep a
# finite but comfortably larger limit for library and production calls.
DEFAULT_MAX_INFERENCE_ITERATIONS = 100000


class ReducedIntegrationError(RuntimeError):
    """A failed reduced solve with the unmodified SciPy result attached."""

    def __init__(self, message, result, diagnostics):
        super().__init__(message)
        self.result = result
        self.diagnostics = diagnostics


def reduced_integration_diagnostics(result, requested_final_time):
    """Summarize a returned ``solve_ivp`` result, including failed results."""
    times = np.asarray(getattr(result, "t", np.array([])), dtype=float)
    state = np.asarray(getattr(result, "y", np.array([])))
    first_nonfinite = None
    if state.size:
        bad = np.argwhere(~np.isfinite(state))
        if bad.size:
            row, column = (int(value) for value in bad[0])
            first_nonfinite = {
                "state_index": row,
                "output_index": column,
                "value": repr(state[row, column]),
            }
    finite_columns = None
    final_norm = None
    maximum_norm = None
    if state.ndim == 2 and state.shape[1]:
        column_norms = np.linalg.norm(state, axis=0)
        finite_columns = bool(np.all(np.isfinite(column_norms)))
        if finite_columns:
            final_norm = float(column_norms[-1])
            maximum_norm = float(np.max(column_norms))
    minimum_separation = None
    if times.size > 1:
        minimum_separation = float(np.min(np.diff(times)))
    return {
        "success": bool(getattr(result, "success", False)),
        "message": str(getattr(result, "message", "no solver message")),
        "last_returned_time": float(times[-1]) if times.size else None,
        "requested_final_time": float(requested_final_time),
        "returned_output_points": int(times.size),
        "nfev": int(getattr(result, "nfev", 0)),
        "njev": int(getattr(result, "njev", 0)),
        "nlu": int(getattr(result, "nlu", 0)),
        "final_latent_state_norm": final_norm,
        "maximum_latent_state_norm": maximum_norm,
        "latent_column_norms_finite": finite_columns,
        "first_nonfinite_value": first_nonfinite,
        "minimum_returned_time_separation": minimum_separation,
        "minimum_internal_time_separation": None,
        "internal_step_history_available": False,
    }


def normalize_embedding_type(value):
    """Normalize public embedding names while retaining historical aliases."""
    if value is None:
        return None
    try:
        return _EMBEDDING_ALIASES[value]
    except (KeyError, TypeError):
        raise ValueError(
            "nonlinear_embedding_type must be None or one of the canonical "
            "values 'elementwise', 'tensorial', 'rbf'; historical aliases "
            "'poly' and 'tens' are also accepted"
        ) from None


def quadratic_features(matrix, embedding_type):
    """Evaluate the current element-wise or non-redundant tensorial features."""
    embedding_type = normalize_embedding_type(embedding_type)
    matrix = np.asarray(matrix)
    if embedding_type == "elementwise":
        return np.power(matrix, 2)
    if embedding_type == "tensorial":
        sym_indices = np.triu_indices(matrix.shape[0])
        products = np.einsum("i...,j...->ij...", matrix, matrix)
        return products[sym_indices[0], sym_indices[1], ...]
    raise ValueError("quadratic features require 'elementwise' or 'tensorial'")


def partition_time_indices(time_steps, training_end_time):
    """Partition an increasing FOM time grid at an inclusive training endpoint."""
    return _partition_time_indices(time_steps, training_end_time)


def _ensure_production_context():
    """Refresh this module's historical aliases from the initialized FOM driver."""
    globals().update(transport_driver.initialize_production_problem())


class NonlinearManifoldReducedModel:
    """
    Nonlinear Manifold Reduced Model for the Sp-DG1 neutron transport solver.
 
    Parameters
    ----------
    nonlinear_embedding_type : str
    ...

    """


    @property
    def nonlinear_embedding_type(self):
        return self._nonlinear_embedding_type


    @nonlinear_embedding_type.setter
    def nonlinear_embedding_type(self, value):
        self._nonlinear_embedding_type = normalize_embedding_type(value)


    def __init__(
        self, 
        nonlinear_embedding_type: str = "tensorial", # "poly", "tens", "rbf", or None
        eps_rbf: float = None,
        every_rbf: int = None,
    ):

        # ── Nonlinear embedding type ───────────────────────────────────────
        self.nonlinear_embedding_type = nonlinear_embedding_type 
        self.nonlinear_function = lambda matrix: None  # Placeholder for the nonlinear function handle (defined in build_nonlinear_bases)

        # ── RBF parameters ─────────────────────────────────────────────────
        self.eps_rbf = eps_rbf      # RBF correlation length
        self.every_rbf = every_rbf  # RBF evaluation frequency (e.g., every 10th snapshot)

        # ── Placeholders – Regularisation weights ──────────────────────────
        self.lambda_E = None       # Embedding regularisation weight
        self.lambda_A = None       # Linear operator regularisation weight
        self.lambda_H = None       # Nonlinear operator regularisation weight

        # ── Placeholders – Simulation parameters ───────────────────────────
        self.TT = None
        self.dt = None
        self.time_steps = None

        # ── Placeholders – Training data parameters ─────────────────────────
        self.solution_path = None
        self.train_fraction = None
        self.training_end_time = None
        self.train_size = None
        self.training_indices = None
        self.extrapolation_indices = None
        self.n_dofs = None

        # ── Placeholders – populated by the run methods ────────────────────
        self.solutionDG1 = None
        self.solutionInf = None
        self.global_training_set = None
        self.global_derivative_set = None

        # ── Placeholders – Reduced-order sizes ─────────────────────────────
        self.size_R = None
        self.size_Q = None

        # ── Placeholders – POD results ─────────────────────────────────────
        self.basis = None
        self.svd_val = None
        self.coefficients = None
 
        # ── Placeholders – Sub-bases ───────────────────────────────────────
        self.pod_linear_basis = None
        self.pod_linear_coeff = None
        self.pod_ortho_basis = None
        self.pod_ortho_coeff = None
        self.pod_global_basis = None
        self.pod_global_coeff = None

        # ── Placeholders – Nonlinear closure bases ─────────────────────────
        self.nonlinear_lift_matrix = None
        self.pod_nonlinear_coeff = None  # evaluated nonlinear features on training set
        self.pod_nonlinear_basis  = None

        # ── Placeholders – Projected linear operators ───────────────────────
        self.projectedDerivativeLinear = None
        self.projectedAbsorptionLinear = None
        self.projectedScatteringLinear = None
        self.projectedStreamingLinear = None
        self.projectedLinear = None

        # ── Placeholders – Projected nonlinear operators ────────────────────
        self.projectedAbsorptionNonlinear = None
        self.projectedScatteringNonlinear = None
        self.projectedStreamingNonlinear = None
        self.projectedNonlinear = None

        # ── Placeholders – Inferred operators ───────────────────────────────
        self.inferredStreamingLinear = None
        self.inferredStreamingNonlinear = None
        self.inferredLinear = None
        self.inferredNonlinear = None
        self.inference_diagnostics = None

        # ── Placeholders – Initial condition ─v──────────────────────────────
        self.initial_condition = None
        self.last_integration_result = None
        self.last_integration_diagnostics = None
        self.last_solve_ivp_elapsed_seconds = None


    # ======================================================================
    # 1. LOAD TRAINING DATA
    # ======================================================================
    def load_training_data(
        self, 
        solution_path: str = SOLUTION_PATH, 
        train_fraction: float = 0.75,
        TT: float = 10.0,
        dt: float = 0.001,
        training_end_time: float = None,
        evaluation_times=None,
    ):
        """Load Sp-DG1 snapshots, compute asymptotic solution, and build training set."""

        _ensure_production_context()

        # Store simulation parameters:
        self.solution_path = solution_path
        self.TT = TT
        self.dt = dt
        
        # Load and validate the phase-space-by-time snapshot array before use.
        snapshots = np.asarray(np.load(self.solution_path))
        if snapshots.ndim != 2:
            raise ValueError(
                f"FOM snapshot array must have rank 2; received rank {snapshots.ndim}"
            )
        expected_phase_rows = globalFF.shape[0]
        if snapshots.shape[0] != expected_phase_rows:
            raise ValueError(
                "FOM snapshot array has the wrong number of phase-space rows: "
                f"expected {expected_phase_rows}, received {snapshots.shape[0]}"
            )
        if not np.all(np.isfinite(snapshots)):
            raise ValueError("FOM snapshot array must contain only finite values")
        self.solutionDG1 = snapshots

        # Use the same inclusive evaluation-time grid as the FOM.
        if evaluation_times is None:
            evaluation_times = transport_driver.make_uniform_time_grid(self.TT, self.dt)
        self.time_steps = np.asarray(evaluation_times, dtype=float)
        if self.solutionDG1.shape[1] != self.time_steps.size:
            raise ValueError(
                "FOM snapshot array has the wrong number of time columns: "
                f"expected {self.time_steps.size}, received {self.solutionDG1.shape[1]}"
            )
        if not np.allclose(
            np.diff(self.time_steps), self.dt, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("FOM evaluation times are inconsistent with dt")
        if not np.isclose(self.time_steps[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
            self.time_steps[-1], self.TT, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("FOM evaluation times must span [0, TT]")

        # Compute the asymptotic solution only after the snapshot is validated.
        self.solutionInf = sparse.linalg.spsolve(globalFF, globalRB)

        # Select training snapshots from the physical time array, including its endpoint.
        self.train_fraction = train_fraction
        if training_end_time is None:
            training_end_time = self.TT * self.train_fraction
        self.training_end_time = float(training_end_time)
        self.training_indices, self.extrapolation_indices = partition_time_indices(
            self.time_steps, self.training_end_time
        )
        self.train_size = self.training_indices.size
        self.n_dofs = self.solutionDG1.shape[0]

        # Subtract asymptotic solution from snapshots to build the training set:
        self.global_training_set = self.solutionDG1[:, self.training_indices] - np.tile(
            self.solutionInf, (self.train_size, 1)
        ).T

        # Print training set shape:
        print(f"Training set shape: {self.global_training_set.shape}")


    # ======================================================================
    # 2. NUMERICALLY APPROXIMATE TIME DERIVATIVES
    # ======================================================================
    def compute_time_derivatives(self):
        """Replace exact time derivatives with 8th-order finite-difference estimates."""

        # Define the coefficients for the 8th-order finite difference scheme:
        c_coeff = np.array([   1/280, -4/105,   1/5 , - 4/5,   0  ,   4/5,  -1/5, 4/105, -  1/280]) / self.dt
        f_coeff = np.array([-761/280,  8    , -14   ,  56/3, -35/2,  56/5, -14/3, 8/7  , -  1/8  ]) / self.dt
        b_coeff = np.array([   1/8  , -8/7  ,  14/3 , -56/5,  35/2, -56/3,  14  ,-8    ,  761/280]) / self.dt
 
        # Initialize the global derivative set:
        self.global_derivative_set = np.zeros(self.global_training_set.shape)
 
        # Compute time derivatives for the first and last 4 snapshots using forward and backward finite differences:
        NN = self.train_size
        if NN < 9:
            raise ValueError("at least nine training snapshots are required for eighth-order derivatives")
        for ii in range(4):
            self.global_derivative_set[:, ii] = self.global_training_set[:, ii : ii + 9] @ f_coeff
            backward_index = NN - 4 + ii
            self.global_derivative_set[:, backward_index] = (
                self.global_training_set[:, backward_index - 8 : backward_index + 1]
                @ b_coeff
            )
        
        # Compute time derivatives for the remaining snapshots using central finite differences:
        for ii in range(NN - 8):
            self.global_derivative_set[:, ii + 4] = self.global_training_set[:, ii : ii + 9] @ c_coeff
 
        # Print confirmation message:
        print("Time derivatives computed (8th-order FD).")


    # ======================================================================
    # 3. PERFORM POD ON THE TRAINING SET
    # ======================================================================
    def compute_pod(self, size_R: int = 16, size_Q: int = 364):
        """
        Compute POD / SVD and split into linear and orthogonal complement bases.
 
        Parameters
        ----------
        size_R : int
            Number of linear POD modes retained (default 16).
        size_Q : int
            Number of additional modes for the nonlinear closure (default 364 = 380 - 16).
        """

        # Compute the SVD of the training set weighted by the square root of the mass matrix:
        basis, svd_val, coefficients = np.linalg.svd(
            globalMMsqrt @ self.global_training_set,
            full_matrices=False,
            compute_uv=True,
            hermitian=False,
        )

        # Store the specified sizes of the linear and orthogonal complement sub-bases:
        self.size_R = size_R
        self.size_Q = size_Q

        # Normalize the basis by the square root of the mass matrix:
        self.basis = linalg.spsolve(globalMMsqrt, basis)
        self.svd_val = svd_val
        self.coefficients = np.diag(svd_val) @ coefficients
 
        # Split the POD basis and coefficients into linear and orthogonal complement sub-bases:
        R, Q = self.size_R, self.size_Q
        self.pod_linear_basis = self.basis[:, :R]
        self.pod_linear_coeff = self.coefficients[:R, :]
        self.pod_ortho_basis  = self.basis[:, R : Q + R]
        self.pod_ortho_coeff  = self.coefficients[R : Q + R, :]
        self.pod_global_basis  = self.basis[:, : Q + R]
        self.pod_global_coeff  = self.coefficients[: Q + R, :]

        # Print confirmation message:
        print(f"POD done. size_R={R}, size_Q={Q}.")


    # ======================================================================
    # 4. COMPUTE NONLINEAR EMBEDDING
    # ======================================================================
    def compute_nonlinear_embedding(self, lambda_E: float = 1e-7):
        """
        Construct polynomial, tensorial, and RBF nonlinear function handles,
        evaluate them on the linear POD coefficients, and compute lifting
        matrices that map nonlinear coefficients to the orthogonal complement.
 
        Parameters
        ----------
        lambda_E : float
            Regularisation weight for the nonlinear embedding (default 1e-7).
        """

        # Store the regularisation weight for the nonlinear embedding:
        self.lambda_E = lambda_E

        # Define nonlinear function handles using normalized public names.
        if self.nonlinear_embedding_type in {"tensorial", "elementwise"}:
            self.nonlinear_function = lambda matrix: quadratic_features(
                matrix, self.nonlinear_embedding_type
            )

        # Define nonlinear function handles based on the specified embedding type (RBF):
        if self.nonlinear_embedding_type == "rbf":
            kernel = lambda xx, x0: np.exp(-(self.eps_rbf * np.linalg.norm((xx.T - x0[:, ...].T).T, axis=0)) ** 2)
            self.nonlinear_function = lambda matrix: np.array(
                [kernel(matrix, self.pod_linear_coeff[:, ii]) for ii in range(0, self.pod_linear_coeff.shape[1], self.every_rbf)]
            )

        # Evaluate the nonlinear function on the linear POD coefficients and compute the nonlinear lift matrix:
        if self.nonlinear_embedding_type is not None:
            pod_nonlinear_coeff = self.nonlinear_function(self.pod_linear_coeff)
            pod_nonlinear_outer = pod_nonlinear_coeff @ pod_nonlinear_coeff.T + self.lambda_E * np.eye(pod_nonlinear_coeff.shape[0])
            self.nonlinear_lift_matrix = np.linalg.solve(pod_nonlinear_outer, pod_nonlinear_coeff).dot(self.pod_ortho_coeff.T).T
            
            # Compute the nonlinear POD basis by applying the nonlinear lift matrix to the orthogonal complement basis:
            self.pod_nonlinear_coeff = pod_nonlinear_coeff.copy()
            self.pod_nonlinear_basis = self.pod_ortho_basis @ self.nonlinear_lift_matrix
        
        # Print a message if no nonlinear embedding type is specified:
        else:
            print("No nonlinear embedding type specified. Skipping nonlinear basis construction.")

        # Print confirmation message:
        print("Nonlinear closure bases built.")

 
    # ======================================================================
    # 5. PROJECT OPERATORS
    # ======================================================================
    def compute_projected_operators(self):
        """Project the full-order FEM operators onto the reduced bases."""
        
        # Project the time derivative snapshots onto the linear basis:
        self.projectedDerivativeLinear  = self.pod_linear_basis.T @ globalMM @ self.global_derivative_set
 
        # Project the absorption and scattering operators onto the linear basis:
        self.projectedAbsorptionLinear  = self.pod_linear_basis.T @ globalAbsorption @ self.pod_linear_basis
        self.projectedScatteringLinear  = self.pod_linear_basis.T @ globalScattering @ self.pod_linear_basis

        # Project the absorption and scattering operators onto the nonlinear basis if a nonlinear embedding type is specified:
        if self.nonlinear_embedding_type is not None:
            self.projectedAbsorptionNonlinear  = self.pod_linear_basis.T @ globalAbsorption @ self.pod_nonlinear_basis
            self.projectedScatteringNonlinear  = self.pod_linear_basis.T @ globalScattering @ self.pod_nonlinear_basis
    
        # Project the full-order streaming operator onto the linear and nonlinear bases:
        try:
            self.projectedStreamingLinear = self.pod_linear_basis.T @ globalStreaming @ self.pod_linear_basis
            if self.nonlinear_embedding_type is not None:
                self.projectedStreamingNonlinear = self.pod_linear_basis.T @ globalStreaming @ self.pod_nonlinear_basis
        except:
            print("Streaming operator not available for projection.")

        # Assemble the projected linear and nonlinear operators (if specified):
        self.projectedLinear = self.projectedStreamingLinear + self.projectedAbsorptionLinear - self.projectedScatteringLinear
        if self.nonlinear_embedding_type is not None:
            self.projectedNonlinear = self.projectedStreamingNonlinear + self.projectedAbsorptionNonlinear - self.projectedScatteringNonlinear

        # Print confirmation message:
        print("Projected operators assembled.")


    # ======================================================================
    # 6. INFER OPERATORS 
    # ======================================================================
    @staticmethod
    def linear_inference(linear_coeff, residual, ll_A=0):
        """
        Linear Operator Inference: recover the linear reduced streaming operator
        from snapshot data via regularised least-squares regression.
        """

        # Compute the regularised pseudo-inverse of the linear coefficients:
        RHS_linear = np.linalg.solve(
            linear_coeff @ linear_coeff.T + ll_A * np.eye(linear_coeff.shape[0]),
            linear_coeff,
        ).T

        # Compute the inferred linear streaming operator:
        inferredStreamingLinear = -residual @ RHS_linear
        return inferredStreamingLinear


    @staticmethod
    def nonlinear_inference(
        linear_coeff,
        nonlinear_coeff,
        residual,
        ll_A=0,
        ll_H=1e-5,
        tolerance=1e-6,
        max_iterations=DEFAULT_MAX_INFERENCE_ITERATIONS,
        return_diagnostics=False,
    ):
        """
        Nonlinear Inference: recover the linear and nonlinear reduced streaming
        operators from snapshot data via an iterative fixed-point scheme.
        """

        # Compute the regularised pseudo-inverses of the linear and nonlinear coefficients:
        RHS_linear    = np.linalg.solve(
            linear_coeff    @ linear_coeff.T    + ll_A * np.eye(linear_coeff.shape[0]),
            linear_coeff,
        ).T
        RHS_nonlinear = np.linalg.solve(
            nonlinear_coeff @ nonlinear_coeff.T + ll_H * np.eye(nonlinear_coeff.shape[0]),
            nonlinear_coeff,
        ).T
 
        # Initialize the inferred linear and nonlinear streaming operators:
        simple_lin    = -residual        @ RHS_linear
        simple_nonlin = -residual        @ RHS_nonlinear
        nl_lin        = -nonlinear_coeff @ RHS_linear
        lin_nl        = -linear_coeff    @ RHS_nonlinear
 
        if max_iterations < 1:
            raise ValueError("max_iterations must be at least one")

        # Initialize the inferred operators and perform the existing alternating updates:
        AA = np.zeros((linear_coeff.shape[0], linear_coeff.shape[0]))
        HH = np.zeros((linear_coeff.shape[0], nonlinear_coeff.shape[0]))
        converged = False
        termination_reason = "maximum_iterations"
        final_convergence_measure = np.inf
        iteration_count = 0
        for iteration_count in range(1, max_iterations + 1):

            # Update the inferred linear and nonlinear streaming operators:
            with np.errstate(over="ignore", invalid="ignore"):
                AA_new = simple_lin    + HH     @ nl_lin
                HH_new = simple_nonlin + AA_new @ lin_nl

            if not np.all(np.isfinite(AA_new)) or not np.all(np.isfinite(HH_new)):
                AA, HH = AA_new, HH_new
                termination_reason = "nonfinite_iterate"
                break

            # Compute the relative changes in the inferred operators and check for convergence:
            err_A = np.linalg.norm(AA_new - AA) / (np.linalg.norm(AA_new) + np.finfo(float).eps)
            err_H = np.linalg.norm(HH_new - HH) / (np.linalg.norm(HH_new) + np.finfo(float).eps)
            final_convergence_measure = float(np.sqrt(err_A ** 2 + err_H ** 2))

            if not np.isfinite(final_convergence_measure):
                AA, HH = AA_new, HH_new
                termination_reason = "nonfinite_convergence_measure"
                break

            # Update the inferred operators for the next iteration:
            AA, HH = AA_new.copy(), HH_new.copy()

            # Check for convergence:
            if final_convergence_measure < tolerance:
                converged = True
                termination_reason = "converged"
                break

        diagnostics = {
            "converged": converged,
            "iteration_count": iteration_count,
            "final_convergence_measure": final_convergence_measure,
            "termination_reason": termination_reason,
        }
        if return_diagnostics:
            return AA, HH, diagnostics
        if not converged:
            raise RuntimeError(
                "nonlinear operator inference did not converge: "
                f"{termination_reason}"
            )
        return AA, HH


    def compute_inferred_operators(
        self,
        lambda_A: float = 0.0,
        lambda_H: float = 1e-3,
        tolerance: float = 1e-6,
        max_iterations: int = DEFAULT_MAX_INFERENCE_ITERATIONS,
    ):
        """
        Use operator inference to approximate the linear and nonlinear streaming operators.
 
        Parameters
        ----------
        lambda_A : float
            Regularisation weight for the linear streaming operator (default 0).
        lambda_H : float
            Regularisation weight for the nonlinear streaming operator (default 1e-3).
        """

        # Store the regularisation weights for the inferred operators:
        self.lambda_A = lambda_A
        self.lambda_H = lambda_H

        # Compute the residual of the projected linear and nonlinear operators:
        resid_linear = self.projectedDerivativeLinear + (self.projectedAbsorptionLinear - self.projectedScatteringLinear) @ self.pod_linear_coeff
        if self.nonlinear_embedding_type is not None:
            resid_nonlinear = resid_linear + (self.projectedAbsorptionNonlinear - self.projectedScatteringNonlinear) @ self.pod_nonlinear_coeff

        # Compute the inferred linear and nonlinear streaming operators using operator inference:
        self.inferredStreamingLinear = self.linear_inference(self.pod_linear_coeff, resid_linear, ll_A=self.lambda_A)
        if self.nonlinear_embedding_type is not None:
            (
                self.inferredStreamingLinear,
                self.inferredStreamingNonlinear,
                self.inference_diagnostics,
            ) = self.nonlinear_inference(
                self.pod_linear_coeff,
                self.pod_nonlinear_coeff,
                resid_nonlinear,
                ll_A=self.lambda_A,
                ll_H=self.lambda_H,
                tolerance=tolerance,
                max_iterations=max_iterations,
                return_diagnostics=True,
            )
            if not self.inference_diagnostics["converged"]:
                reason = self.inference_diagnostics["termination_reason"]
                raise RuntimeError(
                    f"nonlinear operator inference did not converge: {reason}"
                )
            
        # Assemble the inferred linear and nonlinear operators (if specified):
        self.inferredLinear = self.inferredStreamingLinear + self.projectedAbsorptionLinear - self.projectedScatteringLinear
        if self.nonlinear_embedding_type is not None:
            self.inferredNonlinear = self.inferredStreamingNonlinear + self.projectedAbsorptionNonlinear - self.projectedScatteringNonlinear

        # Print confirmation message:
        print("Inferred operators computed.")


    # ======================================================================
    # 7. COMPUTE INITIAL CONDITIONS
    # ======================================================================
    def compute_initial_conditions(self):
        """Return the linear and optimal nonlinear initial conditions."""

        # Define the error function for the linear initial condition:
        self.initial_condition = self.pod_linear_coeff[:, 0] 

        # Define the error function for the nonlinear initial condition:
        if self.nonlinear_embedding_type is not None:
            nonlinear_error = lambda xx: np.linalg.norm(self.pod_global_coeff[:, 0] - np.concatenate((xx, self.nonlinear_lift_matrix @ self.nonlinear_function(xx))))
            self.initial_condition = sp.optimize.minimize(nonlinear_error, self.pod_linear_coeff[:, 0] , method="Nelder-Mead", tol=1e-7, options={"maxiter": 10000})["x"]


    def integrate_reduced(
        self,
        intrusive: bool = True,
        method: str = "Radau",
        atol: float = 1e-12,
        rtol: float = 1e-9,
        initial_time: float = 0.0,
    ):
        """Integrate reduced coefficients using the existing ROM equations."""
        # Compute the initial conditions if they have not been computed yet:
        if self.initial_condition is None:
            self.compute_initial_conditions()

        # Define the right-hand side function for the time integration of the intrusive reduced models:
        if intrusive:
            ff = lambda tt, ss: -self.projectedLinear @ ss
            if self.nonlinear_embedding_type is not None:
                ff = lambda tt, ss: -self.projectedLinear @ ss - self.projectedNonlinear @ self.nonlinear_function(ss)

        # Define the right-hand side function for the time integration of the semi-intrusive reduced models:
        if not intrusive:
            ff = lambda tt, ss: -self.inferredLinear @ ss
            if self.nonlinear_embedding_type is not None:
                ff = lambda tt, ss: -self.inferredLinear @ ss - self.inferredNonlinear @ self.nonlinear_function(ss)

        # Integrate the reduced-order model over the full time interval using a stiff ODE solver:
        ivp_kw = dict(t_span=(initial_time, self.TT), method=method, atol=atol, rtol=rtol, t_eval=self.time_steps)
        solve_ivp_started = time.perf_counter()
        ivp_result = sp.integrate.solve_ivp(
            fun=ff,
            y0=self.initial_condition,
            **ivp_kw,
        )
        self.last_solve_ivp_elapsed_seconds = time.perf_counter() - solve_ivp_started
        self.last_integration_result = ivp_result
        self.last_integration_diagnostics = reduced_integration_diagnostics(
            ivp_result, self.TT
        )
        try:
            validate_solve_ivp_result(
                ivp_result,
                self.time_steps,
                self.initial_condition.size,
                "reduced-order model solve",
                expected_final_time=self.TT,
            )
        except RuntimeError as error:
            raise ReducedIntegrationError(
                str(error), ivp_result, self.last_integration_diagnostics
            ) from error
        return ivp_result


    def reconstruct(self, reduced_coefficients):
        """Reconstruct full states using the existing affine manifold formula."""
        reduced_coefficients = np.asarray(reduced_coefficients)

        # Reconstruct the full-order solution from the reduced coefficients and the POD bases:
        reconstruction = self.solutionInf[:, None] + self.pod_linear_basis @ reduced_coefficients
        if self.nonlinear_embedding_type is not None:
            reconstruction += self.pod_nonlinear_basis @ self.nonlinear_function(reduced_coefficients)

        # Return the reconstructed solution:
        return reconstruction


    # ======================================================================
    # 8. SOLVE REDUCED PROBLEMS
    # ======================================================================
    def solve(self, intrusive: bool = True):
        """
        Integrate the projected or inferred linear, or nonlinear reduced-order models in time
        and reconstruct the full-order solution from the reduced coefficients and the POD bases.
        The `intrusive` flag determines whether to solve the projected (intrusive) or inferred
        (semi-intrusive) reduced models.
        """
        ivp_result = self.integrate_reduced(intrusive=intrusive)
        return self.reconstruct(ivp_result.y)


    # ======================================================================
    # 9. COMPUTE ERRORS
    # ======================================================================
    def compute_errors(self, reconstruction):
        """
        Compute normalised H(MM)-norm errors over the full time horizon.
        """

        # Compute the normalised (MM)-norm errors between the reconstructed solution and the original Sp-DG1 snapshots:
        difference = reconstruction - self.solutionDG1[:, : reconstruction.shape[1]]
        square_err = np.array([globalMM.dot(difference[:, ii]).dot(difference[:, ii]) for ii in range(reconstruction.shape[1])])
        square_inf = globalMM.dot(self.solutionInf).dot(self.solutionInf)

        # Return the normalised (MM)-norm errors:
        return np.sqrt(square_err / square_inf)



# ==============================================================================
# TESTING / ENTRY POINT
# ==============================================================================
def main():
    """Run the existing six-model production ROM workflow explicitly."""
    import time

    if not os.path.exists(SOLUTION_PATH):
        transport_driver.main()
    _ensure_production_context()
 
    print("=" * 70)
    print("NonlinearManifoldReducedModel - Integration Test")
    print("=" * 70)
 
    # ── Hyperparameters ────────────────────────────────────────────────────────
    solution_path   = SOLUTION_PATH
    TT              = 10.0
    DT              = 0.001
    TRAIN_FRACTION  = 0.75
    SIZE_R          = 16        # number of linear POD modes
    SIZE_Q          = 364       # 380 - SIZE_R extra modes for nonlinear closure
    LAMBDA_E        = 1e-7 /  4 * int(TRAIN_FRACTION * TT / DT)
    LAMBDA_A        = 0.0       
    LAMBDA_H_TENS   = 1e-7 * 16 * int(TRAIN_FRACTION * TT / DT)
    LAMBDA_H_POLY   = 1e-4 * 16 * int(TRAIN_FRACTION * TT / DT)
 
    # ── Helper: print a normalised-error summary ───────────────────────────────
    def print_error_summary(label, errors, train_size):
        print(f"  [{label}]")
        print(f"    Max error (train) : {errors[:train_size].max():.4e}")
        print(f"    Max error (extrap): {errors[train_size:].max():.4e}")
        print(f"    Mean error (all)  : {errors.mean():.4e}")
 
    # ── Load training data and compute time derivatives (shared by all models) ─
    prototype_model = NonlinearManifoldReducedModel(nonlinear_embedding_type=None)
    prototype_model.load_training_data(solution_path=solution_path, train_fraction=TRAIN_FRACTION, TT=TT, dt=DT)
    prototype_model.compute_time_derivatives()
    prototype_model.compute_pod(size_R=SIZE_R, size_Q=SIZE_Q)

    # ==========================================================================
    # TEST 1 – LINEAR INTRUSIVE (projected, no nonlinear embedding)
    # ==========================================================================
    print("\n--- TEST 1: Linear intrusive ROM ---")
    t0 = time.time()
 
    model_lin = prototype_model
    model_lin.nonlinear_embedding_type = None
    model_lin.compute_projected_operators()
    model_lin.compute_initial_conditions()
 
    sol_lin = model_lin.solve(intrusive=True)
    err_lin = model_lin.compute_errors(sol_lin)
    print_error_summary("Linear intrusive", err_lin, model_lin.train_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
 
    # ==========================================================================
    # TEST 2 – TENSORIAL NONLINEAR INTRUSIVE (projected)
    # ==========================================================================
    print("\n--- TEST 2: Tensorial nonlinear intrusive ROM ---")
    t0 = time.time()

    model_tens = prototype_model
    model_tens.nonlinear_embedding_type = "tens"
    model_tens.compute_nonlinear_embedding(lambda_E=LAMBDA_E)
    model_tens.compute_projected_operators()
    model_tens.compute_initial_conditions()
 
    sol_tens = model_tens.solve(intrusive=True)
    err_tens = model_tens.compute_errors(sol_tens)
    print_error_summary("Tensorial intrusive", err_tens, model_tens.train_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
 
    # ==========================================================================
    # TEST 3 – POLYNOMIAL NONLINEAR INTRUSIVE (projected)
    # ==========================================================================
    print("\n--- TEST 3: Polynomial nonlinear intrusive ROM ---")
    t0 = time.time()
 
    model_poly = prototype_model
    model_poly.nonlinear_embedding_type = "poly"
    model_poly.compute_nonlinear_embedding(lambda_E=LAMBDA_E)
    model_poly.compute_projected_operators()
    model_poly.compute_initial_conditions()
 
    sol_poly = model_poly.solve(intrusive=True)
    err_poly = model_poly.compute_errors(sol_poly)
    print_error_summary("Polynomial intrusive", err_poly, model_poly.train_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
 
    # ==========================================================================
    # TEST 4 – LINEAR INFERRED (semi-intrusive operator inference)
    # ==========================================================================
    print("\n--- TEST 4: Linear inferred ROM ---")
    t0 = time.time()
 
    model_inf_lin = prototype_model
    model_inf_lin.nonlinear_embedding_type = None
    model_inf_lin.compute_projected_operators()
    model_inf_lin.compute_inferred_operators(lambda_A=LAMBDA_A, lambda_H=1e-5)
    model_inf_lin.compute_initial_conditions()
 
    sol_inf_lin = model_inf_lin.solve(intrusive=False)
    err_inf_lin = model_inf_lin.compute_errors(sol_inf_lin)
    print_error_summary("Linear inferred", err_inf_lin, model_inf_lin.train_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
 
    # ==========================================================================
    # TEST 5 – TENSORIAL INFERRED (semi-intrusive operator inference)
    # ==========================================================================
    print("\n--- TEST 5: Tensorial nonlinear inferred ROM ---")
    t0 = time.time()
 
    model_inf_tens = prototype_model
    model_inf_tens.nonlinear_embedding_type = "tens"
    model_inf_tens.compute_nonlinear_embedding(lambda_E=LAMBDA_E)
    model_inf_tens.compute_projected_operators()
    model_inf_tens.compute_inferred_operators(lambda_A=LAMBDA_A, lambda_H=LAMBDA_H_TENS)
    model_inf_tens.compute_initial_conditions()
 
    sol_inf_tens = model_inf_tens.solve(intrusive=False)
    err_inf_tens = model_inf_tens.compute_errors(sol_inf_tens)
    print_error_summary("Tensorial inferred", err_inf_tens, model_inf_tens.train_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
 
    # ==========================================================================
    # TEST 6 – POLYNOMIAL INFERRED (semi-intrusive operator inference)
    # ==========================================================================
    print("\n--- TEST 6: Polynomial nonlinear inferred ROM ---")
    t0 = time.time()
 
    model_inf_poly = prototype_model
    model_inf_poly.nonlinear_embedding_type = "poly"
    model_inf_poly.compute_nonlinear_embedding(lambda_E=LAMBDA_E)
    model_inf_poly.compute_projected_operators()
    model_inf_poly.compute_inferred_operators(lambda_A=LAMBDA_A, lambda_H=LAMBDA_H_POLY)
    model_inf_poly.compute_initial_conditions()
 
    sol_inf_poly = model_inf_poly.solve(intrusive=False)
    err_inf_poly = model_inf_poly.compute_errors(sol_inf_poly)
    print_error_summary("Polynomial inferred", err_inf_poly, model_inf_poly.train_size)
    print(f"  Elapsed: {time.time()-t0:.1f}s")
 
    # ==========================================================================
    # SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY – Mean normalised (MM)-norm error over all time steps")
    print("=" * 70)
    print(f"  {'Model':<30}  {'Mean error':>12}  {'Max error':>12}")
    print(f"  {'-'*30}  {'-'*12}  {'-'*12}")
    results = [
        ("Linear intrusive",        err_lin),
        ("Tensorial intrusive",     err_tens),
        ("Polynomial intrusive",    err_poly),
        ("Linear inferred",         err_inf_lin),
        ("Tensorial inferred",      err_inf_tens),
        ("Polynomial inferred",     err_inf_poly),
    ]
    for label, err in results:
        print(f"  {label:<30}  {err.mean():>12.4e}  {err.max():>12.4e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
