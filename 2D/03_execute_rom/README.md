# Stage 3: Execute the 2-D reduced-order models

This directory contains the reduced-order modeling calculation used for the
two-dimensional numerical results in the paper.

The main script is:

```text
Nonlinear_Manifold_ROM_2D.py
```

It starts from the data prepared in `../02_prepare_for_rom/`. The script
computes the POD basis, constructs the linear and nonlinear-manifold reduced
models, infers the reduced streaming operators, integrates the reduced
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

From this directory:

```bash
cd 2D/03_execute_rom
python Nonlinear_Manifold_ROM_2D.py
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

The paper calculation retains 132 POD modes. The main comparison uses

```text
N_r = 16
N_q = 64
```

for the nonlinear models. Additional dimensions are exercised by the
Figure 10 parameter study. :contentReference[oaicite:1]{index=1}

## Tests and figures

The four Boolean switches near the bottom of the script control the paper
calculations:

| Switch | Calculation | Output |
| --- | --- | --- |
| `TEST_1` | Relative unresolved POD energy | `average_approximation_error_16_2d.pdf` |
| `TEST_2` | Inferred-model error histories | `relative_error_inferred_models_16_2d.pdf` |
| `TEST_3` | ROM dimension, accuracy, and online speed-up study | `Projected_Integral_Errors_32_2d.pdf` |
| `TEST_4` | Scalar flux and spatial ROM-error fields | `scalar_flux_and_spatial_model_errors_16_2d.pdf` |

The generated figures are written to:

```text
../run/rom/
```

`TEST_3` is the most expensive test because it evaluates several values of
`N_r` and `N_q` and repeats the reduced solves for timing measurements.
:contentReference[oaicite:2]{index=2}

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