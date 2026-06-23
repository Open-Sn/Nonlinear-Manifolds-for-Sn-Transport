# Nonlinear Manifold Reduced-Order Models for 1-D Neutron Transport

Code repository accompanying the paper:

> **"Learning Low-Rank Neutron Transport Dynamics on Linear Subspaces and Nonlinear Manifolds: A Semi-Intrusive Operator-Inferred Approach"**  
> *[in revision]*

---

## Overview

This repository provides a Python implementation of intrusive and semi-intrusive nonlinear manifold reduced-order models (ROMs) for time-dependent, one-dimensional neutron transport problems solved with a simplified $S_P$ discontinuous Galerkin (Sp-DG1) full-order model (FOM). The code is structured around two main files:

| File | Role |
|---|---|
| `Transport_Driver_Benchmark_1D.py` | Full-order solver and data generator |
| `Nonlinear_Manifold_ROM.py` | ROM training, assembly, and time integration |

The benchmark problem considers a three-region 1-D slab with scattering cross sections, driven by a time-dependent incoming angular flux boundary condition. The FOM solution snapshots are used to train and validate six ROM variants spanning linear and nonlinear manifold representations, as well as intrusive (Galerkin projection) and semi-intrusive (operator inference) approaches.

---

## Repository Structure

```
.
├── FLXSLV.py                            # Flux solver: global matrix and vector assembly routines
├── MESH.py                              # Mesh definition and cell/zone layout
├── AQ.py                                # Angular quadrature: directions and weights
├── Transport_Driver_Benchmark_1D.py     # FOM driver: operators, time integration, snapshot generation
└── Nonlinear_Manifold_ROM.py            # NonlinearManifoldReducedModel class + integration tests
```

---

## Dependencies

The code requires Python 3.8+ and the following packages:

```
numpy
scipy
matplotlib
```

All dependencies are available via `pip` or `conda`:

```bash
pip install numpy scipy matplotlib
```

The user-defined modules `FLXSLV`, `MESH`, and `AQ` must also be available on the Python path (see [Configuration](#configuration)).

---

## Configuration

`Transport_Driver_Benchmark_1D.py` inserts the parent directory onto the path at import time, but since all modules live in the same repository this is handled transparently — no manual path configuration is required.

Key simulation parameters are set near the top of `Transport_Driver_Benchmark_1D.py`:

| Parameter | Default | Description |
|---|---|---|
| `ndir` | `4` | Angular quadrature order (number of directions) |
| `width` | `[1.0, 1.0, 1.0]` | Region widths (cm) |
| `n_ref` | `[250, 250, 250]` | Number of cells per region (total: 750) |
| `TT` | `10` | Final simulation time |
| `dt` | `0.001` | Time step size |

---

## Workflow

### Step 1 — Generate FOM Snapshots

Run the transport driver to produce and save the full-order solution snapshots:

```bash
python Transport_Driver_Benchmark_1D.py
```

The script checks whether a snapshot file already exists (e.g., `solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy`). If it does not, the time-dependent transport equation is integrated using SciPy's `solve_ivp` with the `Radau` method and the solution is saved to disk. If the file is found, the solve is skipped and the existing data is loaded.

**Problem setup:**
- Three-region slab geometry (vacuum | scattering | vacuum), total width 3 cm, 750 DG1 cells
- Cross sections: $\sigma_t = [0, 1, 0]$, $\sigma_s = [0, 0.99, 0]$
- Boundary condition: unit incoming flux applied in the most-grazing left-boundary direction
- Initial condition: smooth sigmoid profile, transient towards a non-trivial steady state

### Step 2 — Train and Evaluate Reduced-Order Models

```bash
python Nonlinear_Manifold_ROM.py
```

Running `Nonlinear_Manifold_ROM.py` as a script executes six integration tests in sequence and prints a summary error table to the terminal.

---

## The `NonlinearManifoldReducedModel` Class

The class is the central object of `Nonlinear_Manifold_ROM.py`. A typical workflow chains the following methods:

```python
from Transport_Driver_Benchmark_1D import *

model = NonlinearManifoldReducedModel(nonlinear_embedding_type="tens")

model.load_training_data(solution_path=SOLUTION_PATH, train_fraction=0.75, TT=10.0, dt=0.001)
model.compute_time_derivatives()
model.compute_pod(size_R=16, size_Q=364)
model.compute_nonlinear_embedding(lambda_E=1e-7)
model.compute_projected_operators()              # for intrusive ROMs
model.compute_inferred_operators(lambda_H=1e-4)  # for semi-intrusive ROMs
model.compute_initial_conditions()

solution = model.solve(intrusive=True)       # or intrusive=False
errors   = model.compute_errors(solution)
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nonlinear_embedding_type` | `str` or `None` | `"tensorial"` | Nonlinear embedding type: `"tens"` (tensorial), `"poly"` (polynomial), `"rbf"` (radial basis functions), or `None` (linear subspace) |
| `eps_rbf` | `float` | `None` | RBF correlation length (required for `"rbf"` embedding) |
| `every_rbf` | `int` | `None` | Snapshot stride for RBF centre selection |

### Public Methods

| # | Method | Description |
|---|---|---|
| 1 | `load_training_data()` | Load Sp-DG1 snapshots, compute the asymptotic (steady-state) solution, and build the training set by subtracting the asymptotic solution from each snapshot |
| 2 | `compute_time_derivatives()` | Estimate time derivatives of the training snapshots using an 8th-order finite-difference scheme (central differences in the interior, one-sided forward/backward near the boundaries) |
| 3 | `compute_pod()` | Compute the mass-matrix-weighted SVD of the training set and split the POD basis into a linear subspace of size `size_R` and an orthogonal complement of size `size_Q` |
| 4 | `compute_nonlinear_embedding()` | Construct the nonlinear feature map (tensorial, polynomial, or RBF) on the linear POD coefficients and compute a regularised lifting matrix that maps nonlinear features to the orthogonal complement |
| 5 | `compute_projected_operators()` | Project the full-order FEM operators (mass, absorption, scattering, streaming) onto the linear and nonlinear reduced bases |
| 6 | `compute_inferred_operators()` | Infer the reduced streaming operators from snapshot data using regularised least-squares operator inference; a fixed-point iteration is used for the coupled linear+nonlinear case |
| 7 | `compute_initial_conditions()` | Project the first snapshot onto the reduced manifold; for nonlinear embeddings, the initial condition is found via Nelder–Mead optimisation |
| 8 | `solve()` | Time-integrate the ROM with SciPy's `Radau` solver and reconstruct the full-order solution from the reduced coefficients and POD bases |
| 9 | `compute_errors()` | Compute the normalised mass-matrix ($\mathcal{H}_{M}$) norm error between the ROM reconstruction and the FOM snapshots over all time steps |

---

## ROM Variants Tested

Six ROM configurations are exercised in the integration test block (`__name__ == "__main__"`):

| Test | Embedding | Approach | Key hyperparameters |
|---|---|---|---|
| 1 | None (linear) | Intrusive (Galerkin projection) | `size_R=16` |
| 2 | Tensorial | Intrusive | `size_R=16`, `size_Q=364`, `lambda_E=1e-7/4 * N_train` |
| 3 | Polynomial | Intrusive | same as above |
| 4 | None (linear) | Semi-intrusive (OpInf) | `size_R=16`, `size_Q=364`, `lambda_E=1e-7/4 * N_train`, `lambda_A=0`, `lambda_H=1e-5` |
| 5 | Tensorial | Semi-intrusive (OpInf) | `size_R=16`, `size_Q=364`, `lambda_E=1e-7/4 * N_train`, `lambda_H=1e-7 * 16 * N_train` |
| 6 | Polynomial | Semi-intrusive (OpInf) | `size_R=16`, `size_Q=364`, `lambda_E=1e-7/4 * N_train`, `lambda_H=1e-4 * 16 * N_train` |

The training set uses 75% of the available snapshots; the remaining 25% serve as an extrapolation test set. A summary table of mean and maximum normalised errors is printed at the end.

---
