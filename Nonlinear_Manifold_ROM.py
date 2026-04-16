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
    from utils import *          # provides globalMM, globalFF, globalRB, etc.
    model = NonlinearManifoldReducedModel()
    model.run()
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import copy as copy
import time as time

# ---------------------------------------------------------------------------
# Import mesh/operator data from the project's utils module.
# utils.py is expected to define: globalMM, globalMMsqrt, globalFF, globalRB,
# globalAbsorption, globalScattering, globalStreaming, xx, ndir, and the
# SOLUTION_PATH constant pointing to the Sp-DG1 solution snapshots.
from Transport_Driver_Benchmark_1D import *
# ---------------------------------------------------------------------------


class NonlinearManifoldReducedModel:
    """
    Nonlinear Manifold Reduced Model for the Sp-DG1 neutron transport solver.
 
    Parameters
    ----------
    nonlinear_embedding_type : str
    ...

    """


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
        self.train_size = None
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

        # ── Placeholders – Initial condition ─v──────────────────────────────
        self.initial_condition = None


    # ======================================================================
    # 1. LOAD TRAINING DATA
    # ======================================================================
    def load_training_data(
        self, 
        solution_path: str = SOLUTION_PATH, 
        train_fraction: float = 0.75,
        TT: float = 10.0,
        dt: float = 0.001,
    ):
        """Load Sp-DG1 snapshots, compute asymptotic solution, and build training set."""

        # Store simulation parameters:
        self.solution_path = solution_path
        self.TT = TT
        self.dt = dt
        
        # Load Sp-DG1 solution snapshots and compute asymptotic solution:
        self.solutionDG1 = np.load(self.solution_path)
        self.solutionInf = sparse.linalg.spsolve(globalFF, globalRB)

        # Define the time steps corresponding to the snapshots:
        self.time_steps = np.linspace(0.0, self.TT - self.dt, self.solutionDG1.shape[1])

        # Define training set and compute its size:
        self.train_fraction = train_fraction
        self.train_size = int(self.solutionDG1.shape[1] * self.train_fraction)
        self.n_dofs = self.solutionDG1.shape[0]

        # Subtract asymptotic solution from snapshots to build the training set:
        self.global_training_set = self.solutionDG1[:, : self.train_size : 1] - np.tile(
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
        for ii in range(4):
            self.global_derivative_set[:, ii] = self.global_training_set[:, ii : ii + 9] @ f_coeff
            self.global_derivative_set[:, NN - 4 + ii] = self.global_training_set[:, NN - 4 - 9 : NN - 4] @ b_coeff
        
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

        #  Define nonlinear function handles based on the specified embedding type (tensorial):
        if self.nonlinear_embedding_type == "tens":
            sym_indeces = np.triu_indices(self.size_R)
            self.nonlinear_function = lambda matrix: (
                np.einsum("i...,j...->ij...", matrix, matrix)[sym_indeces[0], sym_indeces[1], ...]
            )

        # Define nonlinear function handles based on the specified embedding type (RBF):
        if self.nonlinear_embedding_type == "rbf":
            kernel = lambda xx, x0: np.exp(-(self.eps_rbf * np.linalg.norm((xx.T - x0[:, ...].T).T, axis=0)) ** 2)
            self.nonlinear_function = lambda matrix: np.array(
                [kernel(matrix, self.pod_linear_coeff[:, ii]) for ii in range(0, self.pod_linear_coeff.shape[1], self.every_rbf)]
            )

        # Define nonlinear function handles based on the specified embedding type (polynomial):
        if self.nonlinear_embedding_type == "poly":
            self.nonlinear_function = lambda matrix: np.power(matrix, 2)    
 
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
    def nonlinear_inference(linear_coeff, nonlinear_coeff, residual, ll_A=0, ll_H=1e-5, tolerance=1e-6):
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
 
        # Initalize the inferred operators and perform fixed-point iterations until convergence:
        AA = np.zeros((linear_coeff.shape[0], linear_coeff.shape[0]))
        HH = np.zeros((linear_coeff.shape[0], nonlinear_coeff.shape[0]))
        while True:

            # Update the inferred linear and nonlinear streaming operators:
            AA_new = simple_lin    + HH     @ nl_lin
            HH_new = simple_nonlin + AA_new @ lin_nl

            # Compute the relative changes in the inferred operators and check for convergence:
            err_A = np.linalg.norm(AA_new - AA) / (np.linalg.norm(AA_new) + np.finfo(float).eps)
            err_H = np.linalg.norm(HH_new - HH) / (np.linalg.norm(HH_new) + np.finfo(float).eps)

            # Update the inferred operators for the next iteration:
            AA, HH = AA_new.copy(), HH_new.copy()

            # Check for convergence:
            if np.sqrt(err_A ** 2 + err_H ** 2) < tolerance:
                break
 
        return AA, HH


    def compute_inferred_operators(self, lambda_A: float = 0.0, lambda_H: float = 1e-3):
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
            self.inferredStreamingLinear, self.inferredStreamingNonlinear = self.nonlinear_inference(
                self.pod_linear_coeff, self.pod_nonlinear_coeff, resid_nonlinear, ll_A=self.lambda_A, ll_H=self.lambda_H)    
            
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
        ivp_kw = dict(t_span=(0.0, self.TT), method="Radau", atol=1e-12, rtol=1e-9, t_eval=self.time_steps)
        ivp_result  = sp.integrate.solve_ivp(fun=ff, y0=self.initial_condition, **ivp_kw)

        # Reconstruct the full-order solution from the reduced coefficients and the POD bases:
        reconstruction = self.solutionInf[:, None] + self.pod_linear_basis @ ivp_result .y
        if self.nonlinear_embedding_type is not None:
            reconstruction += self.pod_nonlinear_basis @ self.nonlinear_function(ivp_result .y)

        # Return the reconstructed solution:
        return reconstruction


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
if __name__ == "__main__":
    import time
 
    print("=" * 70)
    print("NonlinearManifoldReducedModel - Integration Test")
    print("=" * 70)
 
    # ── Hyperparameters ────────────────────────────────────────────────────────
    SOLUTION_PATH   = SOLUTION_PATH
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
    prototype_model.load_training_data(solution_path=SOLUTION_PATH, train_fraction=TRAIN_FRACTION, TT=TT, dt=DT)
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
 

