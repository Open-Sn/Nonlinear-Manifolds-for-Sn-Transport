# Stage 3: Execute the 2-D reduced-order models

This directory contains the reduced-order modeling calculation used for the
two-dimensional numerical results in the paper.

The main script is:

```text
Nonlinear_Manifold_ROM_2D.py
```

It starts from the data written under `../run/preparation/` by Stage 2. The
script computes the POD basis, constructs the linear and nonlinear-manifold
reduced models, infers the reduced streaming operators, integrates the reduced
systems, evaluates the errors, and generates the paper figures.

## Inputs

Stage 2 must be completed first.

The ROM driver reads:

```text
../run/preparation/snapshots/psi_matrix_centered_fp64.npz
../run/preparation/snapshots/center_col.npy

../run/preparation/full_order_out/ops_Mv.npz
../run/preparation/full_order_out/ops_Mt.npz
../run/preparation/full_order_out/ops_Ms.npz
../run/preparation/full_order_out/ops_Mf.npz

../run/opensn/aflux_3newss_1000_0.vtu
```

The centered snapshot matrix has shape

```text
(698880, 1001)
```

corresponding to 32 angular directions and 21,840 DG spatial degrees of
freedom per direction.

The steady-state angular flux in `center_col.npy` is the state that was
subtracted from the transient snapshots during Stage 2.

## Run

From the repository root:

```bash
cd 2D/03_execute_rom
python3 Nonlinear_Manifold_ROM_2D.py
```

The script uses ordinary research parameters and Boolean test switches near
the bottom of the file. There is no command-line interface.

The complete calculation can take substantial time. In particular, the POD
calculation processes the large snapshot matrix, and the Figure 10 dimension
study repeats many reduced solves for timing measurements.

## Reduced-order model construction

The script performs the following main operations:

1. Load the centered snapshots, steady state, and spatial operators.
2. Compute the mass-weighted POD used by the 2-D calculation.
3. Approximate time derivatives of the POD coefficients with 9-point,
   eighth-order finite differences.
4. Construct linear, polynomial, and tensorial nonlinear-manifold models.
5. Project the known reaction terms.
6. Infer the reduced streaming operators from the snapshot data.
7. Compute the reduced initial conditions.
8. Integrate the reduced models with SciPy `solve_ivp`.
9. Compute the paper error measures and generate the figures.

For angular blocks $u_d$ and $v_d$, the phase-space mass inner product
used by the POD is

$$
\langle u,v\rangle_M
=
\sum_{d=1}^{32} u_d^T M_v v_d .
$$

The nonlinear models lift nonlinear features $h(a)$ of the primary reduced
coordinates into the POD complement:

$$
q(a)=E\,h(a),
\qquad
\psi_{\mathrm{ROM}}
\approx \psi_\infty + U_r a + U_q q(a).
$$

Known reaction terms are projected onto the POD spaces, while the reduced
streaming terms are inferred from the snapshot coefficients and their finite-
difference derivatives.

The instantaneous error used for Figure 9 is the steady-state-normalized mass
error

$$
e_j=
\frac{\lVert\psi_j^{\mathrm{ROM}}-\psi_j^{\mathrm{FOM}}\rVert_M}
{\lVert\psi_\infty\rVert_M},
$$

and Figure 10 uses the corresponding relative space-time mass error,

$$
E_{\mathrm{st}}=
\left(
\frac{\sum_j \lVert\psi_j^{\mathrm{ROM}}-\psi_j^{\mathrm{FOM}}\rVert_M^2}
{\sum_j \lVert\psi_j^{\mathrm{FOM}}\rVert_M^2}
\right)^{1/2}.
$$

The paper calculation retains 132 POD modes. The main comparison uses

```text
N_r = 16
N_q = 64
```

for the nonlinear models. Additional dimensions are exercised by the
Figure 10 parameter study.

## Historical time parameterization

The OpenSn snapshots correspond to physical times through approximately
$t=5$, with nominal Stage 1 `dt=0.005`. The preserved paper ROM uses
`TT=5` and `DT=0.005` internally for coefficient differentiation and reduced
integration, while the paper plots label the physical interval as
$t=0,\ldots,5$. This historical parameterization is retained for the paper
reproduction; no additional physical interpretation is asserted here.

## Tests and figures

The four Boolean switches near the bottom of the script control the paper
calculations:

| Switch | Paper figure | Calculation | Output |
| --- | --- | --- | --- |
| `TEST_1` | Figure 8 | Relative unresolved POD energy | `average_approximation_error_16_2d.pdf` |
| `TEST_2` | Figure 9 | Inferred-model error histories | `relative_error_inferred_models_16_2d.pdf` |
| `TEST_3` | Figure 10 | ROM dimension, accuracy, and online speed-up study | `Projected_Integral_Errors_32_2d.pdf` |
| `TEST_4` | Figure B.11 | Scalar flux and spatial ROM-error fields | `scalar_flux_and_spatial_model_errors_16_2d.pdf` |

The generated figures are written to:

```text
../run/rom/
```

`TEST_3` is the most expensive test because it evaluates several values of
`N_r` and `N_q` and repeats the reduced solves for timing measurements.

For a quicker rerun of only one result, the other Boolean switches can be
temporarily set to `False`. Restore the paper settings before committing
source changes.

## Dependencies

The script uses:

- NumPy
- SciPy
- Matplotlib
- VTK

A recent VTK version may emit a deprecation warning for
`grid.GetCells().GetData()` while producing the spatial-error figure. The
warning does not prevent the figure from being generated.

## Workflow

The complete 2-D reproduction is:

```text
01_run_opensn/
    OpenSn full-order calculation
        |
        v
02_prepare_for_rom/
    mass matrices, spatial operators, centered snapshot matrix
        |
        v
03_execute_rom/
    POD, nonlinear-manifold ROMs, operator inference, figures
```

The directory `../02_prepare_for_rom_legacy/` is retained only as a record of
an earlier preprocessing implementation and is not part of the paper
reproduction workflow.
