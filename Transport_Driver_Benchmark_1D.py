# Import external libraries:
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import copy as copy
import time as time
import sys as sys
import os

# Import user defined libraries:
sys.path.insert(0, "../")
from FLXSLV import FLXSLV
from MESH import MESH
from AQ import AQ


def make_uniform_time_grid(t_final, dt):
    """Return an inclusive uniform time grid after validating ``t_final / dt``."""
    n_intervals = int(round(float(t_final) / float(dt)))
    if n_intervals < 0 or not np.isclose(
        n_intervals * float(dt), float(t_final), rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("t_final must be a nonnegative integer multiple of dt")
    return np.arange(n_intervals + 1, dtype=float) * float(dt)


# Select angular quadrature order:
ndir = 4

# Define region widths and intervals in each region:
width = np.array([1.0, 1.0, 1.0])
n_ref = np.array([250, 250, 250], dtype=int)
n_cells = np.sum(n_ref)

# Define the final time and time step for the time-dependent problem:
TT = 10
dt = 0.001
PRODUCTION_TIME_STEPS = make_uniform_time_grid(TT, dt)
NN = PRODUCTION_TIME_STEPS.size

# Define the canonical path for the inclusive production snapshot array.
SOLUTION_PATH = f"solutionDG1_A4_T{TT}_Nt{NN}_Nx{n_cells}_continuous_bis.npy"

# Define a dictionary with the cross sections (total and scattering) for each region:
myXS = {}
myXS["ng"] = 1
myXS["sigt"] = np.array([0.00, 1.00, 0.00])
myXS["sigs"] = np.array([0.00, 0.99, 0.00])

# Define a dictionary with the source terms in each region and with the b.c. for each direction:
mySRC = {}
mySRC["ng"] = 1
mySRC["qext"] = np.zeros(len(width))
mySRC["psi_bc"] = np.zeros(ndir)

# These globals are populated once below to preserve the historical import contract.
# Import still performs no solve, snapshot load, or file write.
myAQ = None
myMESH = None
myFLX = None
xx = None
directAbsorption = None
globalAbsorption = None
directScattering = None
globalScattering = None
directMM = None
globalMM = None
globalInverseMass = None
globalStreaming = None
gobalBD = None
globalFF = None
globalZZ = None
globalRB = None
globalMMsqrt = None
globalMMinv = None
_PRODUCTION_INITIALIZED = False


def initialize_production_problem():
    """Construct and return the existing production mesh and operators once."""
    global myAQ, myMESH, myFLX, xx
    global directAbsorption, globalAbsorption
    global directScattering, globalScattering
    global directMM, globalMM, globalInverseMass, globalStreaming, gobalBD
    global globalFF, globalZZ, globalRB, globalMMsqrt, globalMMinv
    global _PRODUCTION_INITIALIZED

    if not _PRODUCTION_INITIALIZED:
        myAQ = AQ(ndir)
        myMESH = MESH(width, n_ref)
        myFLX = FLXSLV(myAQ, myMESH, myXS, mySRC)

        xx = np.concatenate(
            [[myMESH.x[ii], myMESH.x[ii]] for ii in range(len(myMESH.x))]
        )[1:-1]

        directAbsorption = myFLX.assemble_global_mass_matrix(myXS["sigt"])
        globalAbsorption = sparse.kron(
            np.eye(ndir), directAbsorption, format="csc"
        )

        directScattering = myFLX.assemble_global_mass_matrix(myXS["sigs"])
        globalScattering = sparse.kron(
            np.tile(myAQ.w_q, (ndir, 1)), directScattering, format="csc"
        )

        directMM = myFLX.assemble_global_mass_matrix(np.ones(len(width)))
        globalMM = sparse.kron(np.eye(ndir), directMM, format="csc")
        globalInverseMass = sparse.kron(
            np.eye(ndir), sparse.linalg.inv(directMM), format="csc"
        )
        globalStreaming, gobalBD = myFLX.assemble_global_grad_matrix(
            np.ones(ndir)
        )

        globalFF = globalStreaming + globalAbsorption - globalScattering
        globalZZ = sparse.csc_matrix(globalMM.shape)
        globalRB = gobalBD.dot(
            np.array([1.0 * (ii == (ndir - 1)) for ii in range(ndir)])
        )

        idx = lambda ii: np.ix_([2 * ii, 2 * ii + 1], [2 * ii, 2 * ii + 1])
        globalMMsqrt = copy.deepcopy(globalMM)
        for ii in range(globalMM.shape[0] // 2):
            globalMMsqrt[idx(ii)] = sp.linalg.fractional_matrix_power(
                globalMM[idx(ii)].todense(), 1 / 2
            )
        globalMMinv = copy.deepcopy(globalMM)
        for ii in range(globalMM.shape[0] // 2):
            globalMMinv[idx(ii)] = np.linalg.inv(globalMM[idx(ii)].todense())

        _PRODUCTION_INITIALIZED = True

    return {
        "myAQ": myAQ,
        "myMESH": myMESH,
        "myFLX": myFLX,
        "xx": xx,
        "directAbsorption": directAbsorption,
        "globalAbsorption": globalAbsorption,
        "directScattering": directScattering,
        "globalScattering": globalScattering,
        "directMM": directMM,
        "globalMM": globalMM,
        "globalInverseMass": globalInverseMass,
        "globalStreaming": globalStreaming,
        "gobalBD": gobalBD,
        "globalFF": globalFF,
        "globalZZ": globalZZ,
        "globalRB": globalRB,
        "globalMMsqrt": globalMMsqrt,
        "globalMMinv": globalMMinv,
    }


# Preserve the historical import contract: production operators and their
# module globals exist immediately, but no solve, data load, or file write runs.
initialize_production_problem()


# =======================================================================
# A. HELPER FUNCTIONS FOR TIME-DEPENDENT PROBLEMS
# =======================================================================

# Define helper functions to create time-dependent boundary conditions:
def make_psi_bc_dir(amp_func, left=None, right=None):

    # Define a function that returns the boundary condition vector for a given time t:
    def psi_bc_dir(t):

        # Initialize the boundary condition vector with zeros:
        vec = np.zeros(ndir)
        amp = amp_func(t)
        half = ndir // 2

        # Set the boundary conditions according to the specified types for the left boundary:
        if left == "isotropic"    : vec[half:] = amp
        if left == "most_grazing" : vec[-1]    = amp
        if left == "most_normal"  : vec[half]  = amp

        # Set the boundary conditions according to the specified types for the right boundary:
        if right == "isotropic"   : vec[:half]    = amp
        if right == "most_grazing": vec[0]        = amp
        if right == "most_normal" : vec[half - 1] = amp

        # Return the boundary condition vector:
        return vec

    # Return the function that computes the boundary condition vector for a given time t:
    return psi_bc_dir

# Define helper functions to assemble the volume and boundary source terms:
def assemble_volume_source(qext_now):
    initialize_production_problem()
    q_ext = myFLX.assemble_global_qext_vector(np.asarray(qext_now) / np.sum(myAQ.w_q))
    return np.kron(np.ones(ndir), q_ext)

# Define helper functions to assemble the volume and boundary source terms:
def assemble_boundary_source(psi_bc_now):
    initialize_production_problem()
    return gobalBD.dot(np.asarray(psi_bc_now))

# Define the right-hand side of the transport equation for time integration:
def transport_rhs(t, Psi_vec, qext_func=None, psi_bc_func=None):

    initialize_production_problem()

    # If the source term functions are not provided, use default functions that return zero sources:
    if qext_func is None  : qext_func   = lambda tt: np.zeros(myMESH.n_zones)
    if psi_bc_func is None: psi_bc_func = lambda tt: np.zeros(ndir)

    # Assemble the volume and boundary source terms for the current time t:
    b_vol = assemble_volume_source(qext_func(t))
    b_bd  = assemble_boundary_source(psi_bc_func(t))

    # Compute the right-hand side of the transport equation using the global operators and source terms:
    return globalInverseMass.dot(b_vol + b_bd - globalFF.dot(Psi_vec))

def validate_solve_ivp_result(
    result,
    requested_times,
    expected_state_size,
    context,
    expected_final_time=None,
):
    """Validate a ``solve_ivp`` result before scientific data are consumed."""
    requested_times = np.asarray(requested_times, dtype=float)
    returned_times = np.asarray(getattr(result, "t", np.array([])), dtype=float)

    if not bool(getattr(result, "success", False)):
        message = getattr(result, "message", "no solver message")
        raise RuntimeError(f"{context}: solver failure: {message}")
    if returned_times.ndim != 1 or returned_times.size != requested_times.size:
        raise RuntimeError(
            f"{context}: incomplete output: expected {requested_times.size} times, "
            f"received {returned_times.size}"
        )
    if not np.allclose(returned_times, requested_times, rtol=0.0, atol=1.0e-12):
        raise RuntimeError(f"{context}: time-grid mismatch")
    if expected_final_time is None:
        expected_final_time = requested_times[-1]
    if returned_times.size and not np.isclose(
        returned_times[-1], expected_final_time, rtol=0.0, atol=1.0e-12
    ):
        raise RuntimeError(f"{context}: time-grid mismatch at final time")

    solution = np.asarray(getattr(result, "y", np.array([])))
    expected_shape = (int(expected_state_size), requested_times.size)
    if solution.shape != expected_shape:
        raise RuntimeError(
            f"{context}: shape mismatch: expected {expected_shape}, "
            f"received {solution.shape}"
        )
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(f"{context}: non-finite values in solution")
    return result


def make_production_initial_condition(dof_coordinates):
    """Construct the preserved localized-sigmoid production initial condition."""
    dof_coordinates = np.asarray(dof_coordinates)
    return np.concatenate(
        [
            0 * dof_coordinates,
            0 * dof_coordinates,
            0 * dof_coordinates,
            1 - 1 / (1 + np.exp(-100 * (dof_coordinates - 0.1))),
        ]
    )


# Define a helper function to solve the time-dependent transport equation using scipy's ODE solver:
def solve_transport(
    Psi0,
    t_final,
    qext_func=None,
    psi_bc_func=None,
    n_output_times=1001,
    method="Radau",
    atol=1e-9,
    rtol=1e-12,
    evaluation_times=None,
):
    initialize_production_problem()

    # Define the function that computes the right-hand side of the transport equation for time integration:
    ff = lambda t, Psi: transport_rhs(t, Psi, qext_func=qext_func, psi_bc_func=psi_bc_func)
    if evaluation_times is None:
        tt = np.linspace(0.0, t_final, n_output_times)
    else:
        tt = np.asarray(evaluation_times, dtype=float)
        if tt.ndim != 1 or tt.size == 0:
            raise ValueError("evaluation_times must be a nonempty one-dimensional array")
        if not np.isclose(tt[0], 0.0, rtol=0.0, atol=1.0e-12) or not np.isclose(
            tt[-1], t_final, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError("evaluation_times must span [0, t_final]")

    # Use scipy's ODE solver to solve the time-dependent transport equation:
    result = sp.integrate.solve_ivp(
        fun=ff,
        t_span=(0.0, t_final),
        y0=np.asarray(Psi0),
        method=method,
        atol=atol,
        rtol=rtol,
        t_eval=tt,
    )
    return validate_solve_ivp_result(
        result,
        tt,
        np.asarray(Psi0).size,
        "full-order transport solve",
        expected_final_time=t_final,
    )


# =======================================================================
# B. SOLVE THE TIME-DEPENDENT PROBLEM AND SAVE THE SOLUTION FOR LATER USE
# =======================================================================

def main():
    """Run the existing production full-order snapshot workflow explicitly."""
    initialize_production_problem()

    # Check if the production snapshot exists and solve only when explicitly run.
    if not os.path.exists(SOLUTION_PATH):
        print(
            f"File '{SOLUTION_PATH}' not found. Solving the time-dependent "
            "transport equation and saving the solution. This may take a while..."
        )

        # Historical label preserved: this selects the most-normal positive ordinate.
        psi_bc_func = make_psi_bc_dir(
            lambda tt: 1.0, left="most_grazing", right=None
        )

        # Preserve the existing localized sigmoid exactly.
        Psi0 = make_production_initial_condition(xx)

        sol = solve_transport(
            Psi0=Psi0,
            t_final=TT,
            qext_func=lambda tt: np.zeros(myMESH.n_zones),
            psi_bc_func=psi_bc_func,
            n_output_times=NN,
            evaluation_times=PRODUCTION_TIME_STEPS,
        )
        np.save(SOLUTION_PATH, sol.y)
    else:
        print(
            f"File '{SOLUTION_PATH}' found. Skipping the time-dependent solve "
            "and loading the solution from the file."
        )


if __name__ == "__main__":
    main()
