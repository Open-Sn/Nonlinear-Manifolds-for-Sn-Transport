# Import external libraries:
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
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

# Select angular quadrature order:
ndir = 4
myAQ = AQ(ndir)

# Define region widths and intervals in each region:
width = np.array([1.0, 1.0, 1.0])
n_ref = np.array([250, 250, 250], dtype=int)
n_cells = np.sum(n_ref)

# Define the final time and time step for the time-dependent problem:
TT = 10
dt = 0.001
NN = int(TT / dt) + 1

# Define the path to save the solution of the time-dependent transport equation:
SOLUTION_PATH = f"solutionDG1_A4_T{TT}_Nt{NN}_Nx{n_cells}_continuous_bis.npy"

# Define the mesh:
myMESH = MESH(width, n_ref)

# Define a dictionary with the cross sections (total and scattering) for each region:
myXS = {}
myXS["ng"] = 1
myXS["sigt"] = np.array([0.00, 1.00, 0.00])
myXS["sigs"] = np.array([0.00, 0.99, 0.00])

# Define a dictionary with the source terms in each region and with the b.c. for each direction:
mySRC = {}
mySRC["ng"] = 1
mySRC["qext"] = np.zeros(myMESH.n_zones) 
mySRC["psi_bc"] = np.zeros(ndir)

# Assemble problem solver:
myFLX = FLXSLV(myAQ, myMESH, myXS, mySRC)

# Define coordinates of the degrees of freedom:
xx = np.concatenate([[myMESH.x[ii], myMESH.x[ii]] for ii in range(len(myMESH.x))])
xx = xx[1:-1]

# Define monolithic absorption matrix:
directAbsorption = myFLX.assemble_global_mass_matrix(myXS["sigt"])
globalAbsorption = sparse.kron(np.eye(ndir), directAbsorption, format="csc")

# Define monolithic scattering matrix:
directScattering = myFLX.assemble_global_mass_matrix(myXS["sigs"])
globalScattering = sparse.kron(np.tile(myAQ.w_q, (ndir,1)), directScattering, format="csc")

# Define monolithic mass matrix:
directMM = myFLX.assemble_global_mass_matrix(np.ones(len(width)))
globalMM = sparse.kron(np.eye(ndir), directMM, format="csc")

# Define monolithic inverse mass matrix:
globalInverseMass = sparse.kron(np.eye(ndir), sparse.linalg.inv(directMM), format="csc")
globalStreaming, gobalBD = myFLX.assemble_global_grad_matrix(np.ones(ndir))

# Define monolithic boundary consitions and splu decomposed LHS:
globalFF = globalStreaming + globalAbsorption - globalScattering
globalZZ = sparse.csc_matrix(globalMM.shape)
globalRB = gobalBD.dot(np.array([1.0 * (ii==(ndir-1)) for ii in range(ndir)]))

# Define square root global and direct mass matrix:
idx = lambda ii: np.ix_([2*ii,2*ii+1],[2*ii,2*ii+1])
globalMMsqrt = copy.deepcopy(globalMM)
for ii in range(globalMM.shape[0]//2): globalMMsqrt[idx(ii)] = sp.linalg.fractional_matrix_power(globalMM[idx(ii)].todense(), 1/2)
globalMMinv = copy.deepcopy(globalMM)
for ii in range(globalMM.shape[0]//2): globalMMinv[idx(ii)] = np.linalg.inv(globalMM[idx(ii)].todense())


# =======================================================================
# A. HELPER FUNCTIONS FOR TIME-DEPENDENT PROBLEMS
# =======================================================================

# Define helper functions to create time-dependent boundary conditions:
def make_psi_bc_dir(amp_func, left=None, right=None):

    # Define a function that returns the boundary condition vector for a given time t:
    def psi_bc_dir(t):

        # Initialize the boundary condition vector with zeros:
        vec = np.zeros(myAQ.ndir)
        amp = amp_func(t)
        half = myAQ.ndir // 2

        # Set the boundary conditions according to the specified types for the left boundary:
        if left == "isotropic"    : vec[half:] = amp
        if left == "most_grazing" : vec[half]  = amp
        if left == "most_normal"  : vec[-1]    = amp

        # Set the boundary conditions according to the specified types for the right boundary:
        if right == "isotropic"   : vec[:half]    = amp
        if right == "most_grazing": vec[half - 1] = amp
        if right == "most_normal" : vec[0]        = amp

        # Return the boundary condition vector:
        return vec

    # Return the function that computes the boundary condition vector for a given time t:
    return psi_bc_dir

# Define helper functions to assemble the volume and boundary source terms:
def assemble_volume_source(qext_now):
    q_ext = myFLX.assemble_global_qext_vector(np.asarray(qext_now) / np.sum(myAQ.w_q))
    return np.kron(np.ones(ndir), q_ext)

# Define helper functions to assemble the volume and boundary source terms:
def assemble_boundary_source(psi_bc_now):
    return gobalBD.dot(np.asarray(psi_bc_now))

# Define the right-hand side of the transport equation for time integration:
def transport_rhs(t, Psi_vec, qext_func=None, psi_bc_func=None):

    # If the source term functions are not provided, use default functions that return zero sources:
    if qext_func is None  : qext_func   = lambda tt: np.zeros(myMESH.n_zones)
    if psi_bc_func is None: psi_bc_func = lambda tt: np.zeros(ndir)

    # Assemble the volume and boundary source terms for the current time t:
    b_vol = assemble_volume_source(qext_func(t))
    b_bd  = assemble_boundary_source(psi_bc_func(t))

    # Compute the right-hand side of the transport equation using the global operators and source terms:
    return globalInverseMass.dot(b_vol + b_bd - globalFF.dot(Psi_vec))

# Define a helper function to solve the time-dependent transport equation using scipy's ODE solver:
def solve_transport(Psi0, t_final, qext_func=None, psi_bc_func=None, n_output_times=1001, method="Radau", atol=1e-9, rtol=1e-12):
    
    # Define the function that computes the right-hand side of the transport equation for time integration:
    ff = lambda t, Psi: transport_rhs(t, Psi, qext_func=qext_func, psi_bc_func=psi_bc_func)
    tt = np.linspace(0.0, t_final, n_output_times)

    # Use scipy's ODE solver to solve the time-dependent transport equation:
    return sp.integrate.solve_ivp(fun=ff, t_span=(0.0, t_final), y0=np.asarray(Psi0), method=method, atol=atol, rtol=rtol, t_eval=tt)


# =======================================================================
# B. SOLVE THE TIME-DEPENDENT PROBLEM AND SAVE THE SOLUTION FOR LATER USE
# =======================================================================

# Check if the file "solutionDG1_A4_T10_Nt10000_Nx750_continuous.npy" exists and if not, solve the time-dependent transport equation:
if not os.path.exists(SOLUTION_PATH):
    print(f"File '{SOLUTION_PATH}' not found. Solving the time-dependent transport equation and saving the solution. This may take a while...")

    # Define a boundary condition function that applies a constant incoming flux of 1.0 in the most normal direction:
    psi_bc_func = make_psi_bc_dir(lambda tt: 1.0, left="most_normal", right=None)

    # Define an initial condition with a smooth transition from 0 to 1 in the first region:
    Psi0 = np.concatenate([0 * xx, 0 * xx, 0 * xx, 1 - 1 / (1 + np.exp(-100 * (xx - 0.1)))])

    # Solve the time-dependent transport equation with the specified initial condition and boundary conditions:
    sol = solve_transport(Psi0=Psi0, t_final=TT, qext_func=lambda tt: np.zeros(myMESH.n_zones), 
                          psi_bc_func=psi_bc_func, n_output_times=NN)

    # Save the solution to a file for later use:
    np.save(SOLUTION_PATH, sol.y)

# If the file already exists, skip the time-dependent solve and load the solution from the file:
else: 
    print(f"File '{SOLUTION_PATH}' found. Skipping the time-dependent solve and loading the solution from the file.")
