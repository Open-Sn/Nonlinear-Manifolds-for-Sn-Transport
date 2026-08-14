# Nonlinear-Manifold ROMs for 1-D Neutron Transport

This repository implements a one-dimensional discrete-ordinates neutron-transport benchmark for studying linear and nonlinear-manifold reduced-order models (ROMs). It compares projected/intrusive and inferred/semi-intrusive formulations on linear, polynomial, and tensorial trial manifolds.

The implementation accompanies the paper *Learning Low-Rank Neutron Transport Dynamics on Linear Subspaces and Nonlinear Manifolds: A Semi-Intrusive Operator-Inferred Approach* (in revision).

## Repository layout

| File | Purpose |
|---|---|
| `Transport_Driver_Benchmark_1D.py` | Defines the standard transport problem, assembles the full-order operators, and generates the snapshot matrix when needed. |
| `Nonlinear_Manifold_ROM.py` | Implements the ROM workflow and runs the baseline studies and Figure 1–5 calculations. |
| `PLOT.py` | Contains the plotting routines used to create Figures 1–5. |
| `FLXSLV.py` | Assembles the DG1 transport, total-interaction, scattering, mass, source, and boundary operators. |
| `MESH.py` | Builds the one-dimensional multiregion mesh. |
| `AQ.py` | Constructs the Gauss-Legendre angular quadrature. |
| `README.md` | Describes the benchmark and its reproducible workflow. |

## Dependencies

The standard workflow requires:

- NumPy
- SciPy
- Matplotlib

Seaborn is optional and is used only by an auxiliary flux-plotting routine in `FLXSLV.py`.

One tested environment is:

```text
Python     3.13.9
NumPy      2.3.5
SciPy      1.16.3
Matplotlib 3.10.6
```

These are tested versions, not minimum-version guarantees. A basic installation is:

```powershell
python -m pip install numpy scipy matplotlib
```

## Standard transport benchmark

The benchmark is a one-group problem on three adjacent regions:

| Quantity | Standard value |
|---|---|
| Domain | $0\le x\le 3$ |
| Region widths | `[1, 1, 1]` |
| Spatial mesh | 250 cells per region; 750 cells total |
| Spatial discretization | DG1, two degrees of freedom per cell |
| Angular discretization | Four-point Gauss-Legendre ($S_4$) quadrature |
| Full state dimension | $4\times(2\times750)=6000$ |
| Total cross section | $\Sigma_t=[0,1,0]$ |
| Scattering cross section | $\Sigma_s=[0,0.99,0]$ |
| Volumetric source | Zero |
| Incoming boundary | Constant unit angular flux at the left incoming boundary in the most-normal $S_4$ direction, $\mu\approx0.861136$ |
| Final time | $T=10$ |
| Stored time spacing | $\Delta t=0.001$ |
| Stored snapshots | 10,001 |

The first three angular components are initially zero. The fourth uses the smooth profile
$1-[1+\exp(-100(x-0.1))]^{-1}$. The full-order model (FOM) is integrated with SciPy's `Radau` method using `atol=1e-9` and `rtol=1e-12`.

## Snapshot generation and reuse

The generated snapshot file is

```text
solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy
```

It has shape `(6000, 10001)` and occupies approximately 480 MB. It is ignored by Git and is therefore absent from a fresh clone.

- If the file is absent, `Transport_Driver_Benchmark_1D.py` performs the full FOM solve and writes it.
- If the file exists, the driver skips the FOM solve.
- `Nonlinear_Manifold_ROM.py` imports the transport driver. Running or importing the ROM module can therefore trigger snapshot generation when the file is absent; the ROM subsequently loads the snapshot.

## Running the standard workflow

Run commands from the repository root. First generate or reuse the FOM snapshots:

```powershell
python Transport_Driver_Benchmark_1D.py
```

Then run the standard ROM studies and create Figures 1–5:

```powershell
python Nonlinear_Manifold_ROM.py
```

The ROM script displays plots as well as saving them. In a headless PowerShell environment, use a noninteractive Matplotlib backend:

```powershell
$env:MPLBACKEND='Agg'
python Nonlinear_Manifold_ROM.py
```

The FOM and the ROM parameter sweeps are computationally substantial. Existing snapshots avoid repeating the FOM integration.

## Standard ROM configuration

### Training data and POD spaces

| Quantity | Value |
|---|---:|
| `train_fraction` | 0.75 |
| `train_size` | 7500 |
| Linear dimension $N_r$ | 16 |
| Closure dimension $N_q$ | 548 |
| Total POD dimension $N_r+N_q$ | 564 |

Training uses snapshot indices 0–7499, corresponding to $t=0$ through $t=7.499$. Extrapolation begins at $t=7.5$ and contains 2,501 snapshots through $t=10$.

Snapshots are centered about the computed steady solution before the mass-matrix-weighted POD is formed. Snapshot derivatives used by operator inference are evaluated with eighth-order finite differences.

Supported manifold identifiers are:

| Value | Trial manifold |
|---|---|
| `None` | Linear POD subspace |
| `"tensorial"` | Unique quadratic cross-products of the reduced coordinates |
| `"poly"` | Componentwise quadratic (polynomial) features |
| `"rbf"` | Gaussian radial-basis features |

For projected ROMs, the total-interaction and scattering operators are projected onto the reduced spaces; the streaming operator is either projected for the intrusive formulation or learned from snapshot data for the semi-intrusive formulation. The ROM trajectories use `Radau` with `atol=1e-12` and `rtol=1e-9`.

### Baseline regularization

For Tests 1–6, $N_{\mathrm{train}}=7500$ and the standard values are:

| Model term | Formula | Value |
|---|---:|---:|
| Tensorial embedding $\lambda_E$ | $10^{-7}N_{\mathrm{train}}/4$ | $1.875\times10^{-4}$ |
| Polynomial embedding $\lambda_E$ | $10^{-7}(64)N_{\mathrm{train}}$ | $4.8\times10^{-2}$ |
| Linear inferred operator $\lambda_A$ | $0$ | $0$ |
| Tensorial inferred nonlinear operator $\lambda_H$ | $10^{-7}(16)N_{\mathrm{train}}$ | $1.2\times10^{-2}$ |
| Polynomial inferred nonlinear operator $\lambda_H$ | $10^{-4}(16)N_{\mathrm{train}}$ | $12$ |

The Figure 4 and Figure 5 sweeps use the study-specific regularization schedules defined in the executable script rather than assuming that every sweep point uses this baseline table.

The six baseline variants are the linear subspace and the tensorial and polynomial manifolds, with either projected/intrusive or inferred/semi-intrusive streaming operators.

### Programmatic use

Importing the ROM class also imports the transport driver and may generate the FOM snapshot if it is missing.

```python
from Nonlinear_Manifold_ROM import NonlinearManifoldReducedModel

n_train = 7500
model = NonlinearManifoldReducedModel(nonlinear_embedding_type="tensorial")
model.load_training_data(train_fraction=0.75, TT=10.0, dt=0.001)
model.compute_time_derivatives()
model.compute_pod(size_R=16, size_Q=548)
model.compute_nonlinear_embedding(lambda_E=1e-7 * n_train / 4)
model.compute_projected_operators()
model.compute_inferred_operators(lambda_A=0.0, lambda_H=1e-7 * 16 * n_train)
model.compute_initial_conditions()
reconstruction = model.solve(intrusive=False)
errors = model.compute_errors(reconstruction)
```

## Error and timing measures

Figures 2 and 3 use the normalized mass-matrix error at each stored time,

$$
e_i=\left(
\frac{(u_i^{\mathrm{ROM}}-u_i^{\mathrm{FOM}})^T M
(u_i^{\mathrm{ROM}}-u_i^{\mathrm{FOM}})}
{u_\infty^T M u_\infty}
\right)^{1/2}.
$$

Figures 4 and 5 instead use the aggregate relative space-time error

$$
E_{\mathrm{st}}=\left(
\frac{\sum_i (u_i^{\mathrm{ROM}}-u_i^{\mathrm{FOM}})^T M
(u_i^{\mathrm{ROM}}-u_i^{\mathrm{FOM}})}
{\sum_i (u_i^{\mathrm{FOM}})^T M u_i^{\mathrm{FOM}}}
\right)^{1/2}.
$$

ROM online time measures only the `solve_ivp` integration and excludes full-state reconstruction. Speed-up uses a fixed FOM reference timing retained from the original benchmark/reproduction workflow; absolute timings and speed-ups are machine dependent.

## Generated figures

Figure 1 plots the implemented relative unresolved POD energy. With singular values $\sigma_j$ and one-based plotted index $r$,

$$
\rho_{\mathrm{miss}}(r)=
\left(
\frac{\displaystyle\sum_{j=r}^{N}\sigma_j^2}
{\displaystyle\sum_{j=1}^{N}\sigma_j^2}
\right)^{1/2}.
$$

| Output file | Contents |
|---|---|
| `Fig_01_average_approximation_error_16.pdf` | Relative unresolved POD energy, highlighting the baseline $N_r=16$ and following $N_q=548$ POD modes. |
| `Fig_02_Projected_16.pdf` | Projected/intrusive linear, polynomial, and tensorial solution comparisons and normalized time-dependent errors. |
| `Fig_03_Inferred_16.pdf` | Equivalent comparison for inferred/semi-intrusive ROMs. |
| `Fig_04_Projected_Integral_Errors_d.pdf` | Fixed-total-dimension study using $N_r=[8,16,24,32,40,48,56,64]$ and $N_q=564-N_r$. Tensorial, polynomial, and linear ROMs are compared for projected and inferred streaming operators using aggregate space-time error and online speed-up. |
| `Fig_05_Projected_Integral_Errors_Nq_test.pdf` | Closure study using $N_q=[0,1,2,4,8,16,32,64,128]$. It compares tensorial and polynomial ROMs with $N_r=32$, fixed linear $N_r=32$, expanded linear $N_r=32+N_q$, and a POD projection baseline. $N_q=0$ is computed but omitted from the logarithmic $N_q$ plot. |

The main script labels its stages as follows:

- Tests 1–6: the six baseline projected/inferred ROM variants.
- Tests 7–9: Figures 1–3.
- Test 10: the $N_r$ sweep for Figure 4.
- Test 11: the $N_q$ sweep for Figure 5.

The generated `.npy` and `.pdf` artifacts are ignored by Git.

## License

See `LICENSE`.
