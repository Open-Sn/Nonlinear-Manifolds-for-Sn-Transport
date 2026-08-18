# Stage 2: prepare inputs for the paper ROM

## Purpose

Stage 2 converts the OpenSn angular-flux output into the inputs required by the
paper ROM.

## Input data

The scripts read the steady state and all 1001 transient states from:

```text
../run/opensn/aflux_3newss_1000_0.vtu
../run/opensn/transient/aflux_3newh_0001_0.vtu
...
../run/opensn/transient/aflux_3newh_1001_0.vtu
```

There are 32 angular directions and 21,840 DG spatial degrees of freedom per
direction, so each angular state has 698,880 entries. The steady angular flux
is subtracted from every transient state.

## Sequence

From this directory, run:

```bash
python compute_mass_matrix_from_vtu.py
python build_full_order_ops_direction_indep.py
python build_snapshot_matrix_from_vtu.py
```

## Outputs

Mass matrices are written to:

```text
../run/preparation/dg_mass_out/
```

Full-order spatial operators are written to:

```text
../run/preparation/full_order_out/
```

The centered training data and steady-state centering vector are written to:

```text
../run/preparation/snapshots/psi_matrix_centered_fp64.npz
../run/preparation/snapshots/center_col.npy
```

POD construction is not part of Stage 2. The paper ROM driver in Stage 3
computes the POD itself.
