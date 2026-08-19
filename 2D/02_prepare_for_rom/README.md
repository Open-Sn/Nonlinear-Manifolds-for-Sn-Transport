# Stage 2: prepare inputs for the paper ROM

[Back to the 2-D workflow](../README-2D.md)

## Purpose

Stage 2 converts the OpenSn angular-flux output into the spatial operators and
centered training data required by the paper ROM. It performs three operations:

1. construct the DG spatial mass matrices;
2. construct the direction-independent full-order reaction/mass operators;
3. construct the centered angular-flux snapshot matrix and save its center.

POD construction is not part of Stage 2. The paper ROM driver in Stage 3
computes the mass-weighted POD itself.

## Input data

The scripts read the steady state and all 1001 transient states from:

```text
../run/opensn/aflux_3newss_1000_0.vtu
../run/opensn/transient/aflux_3newh_0001_0.vtu
...
../run/opensn/transient/aflux_3newh_1001_0.vtu
```

For the recommended reproduction path, download `3newh_aflx.tar` and
`aflux_3newss_1000_0.vtu` from
[Zenodo](https://doi.org/10.5281/zenodo.21762243). Place both under
`2D/run/opensn/` and extract the 1001 transient angular-flux VTUs into
`2D/run/opensn/transient/`.

There are 32 angular directions and 21,840 DG spatial degrees of freedom per
direction, so each angular state has 698,880 entries.

## Preprocessing

For transient state $\psi_j$ and steady angular state $\psi_\infty$, the
centered snapshot is

$$
X_j = \psi_j - \psi_\infty .
$$

For each triangular cell $K$, the consistent P1 element mass matrix is

$$
M_K =
\frac{|K|}{12}
\begin{bmatrix}
2 & 1 & 1\\
1 & 2 & 1\\
1 & 1 & 2
\end{bmatrix}.
$$

The cell contributions are collected into one sparse matrix $M_m$ per
material. Material-weighted spatial operators are then assembled, for example,

$$
M_t=\sum_m \sigma_{t,m}M_m,
\qquad
M_v=\sum_m v_m^{-1}M_m,
\qquad
M_s=\sum_m \sigma_{s,m}M_m,
$$

with analogous assembly for the production and external-source operators.

## Dependencies

The three scripts require:

- NumPy
- SciPy
- VTK Python bindings
- Matplotlib

## Sequence

From `2D/02_prepare_for_rom/`, run:

```bash
python3 compute_mass_matrix_from_vtu.py
python3 build_full_order_ops_direction_indep.py
python3 build_snapshot_matrix_from_vtu.py
```

The snapshot calculation creates a compressed matrix of approximately 5.6 GB
before compression and therefore requires substantial time, memory, and disk
space.

## Outputs

DG mass matrices and compact mesh data are written to:

```text
../run/preparation/dg_mass_out/
```

The principal spatial operators needed by Stage 3 are:

```text
../run/preparation/full_order_out/ops_Mv.npz
../run/preparation/full_order_out/ops_Mt.npz
../run/preparation/full_order_out/ops_Ms.npz
../run/preparation/full_order_out/ops_Mf.npz
```

The centered training data and steady-state center are:

```text
../run/preparation/snapshots/psi_matrix_centered_fp64.npz
../run/preparation/snapshots/center_col.npy
```

The NPZ stores the centered snapshot matrix under key `X` with expected shape
`(698880, 1001)`.
