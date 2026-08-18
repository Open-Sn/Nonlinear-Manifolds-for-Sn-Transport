# Stage 2: legacy 2D preprocessing

> **Status.** This folder preserves Jean Ragusa's early 2D preprocessing
> implementation because it contains useful, reproducible numerical and data-
> handling techniques. It is not claimed to be the exact code path used for
> every final paper result. The exact paper-result workflow will be documented
> separately.

The scientific scripts retain their original headers, organization, variable
names, comments, and algorithms. Only paths needed by the present `2D/`
layout, a check for the expected 1001 input snapshots, the requested POD rank,
and the complete-spectrum singular-value plot have been added.

## Running the legacy sequence

Run the scripts from `2D/02_prepare_for_rom_legacy/`:

```bash
python compute_mass_matrix_from_vtu.py
python build_full_order_ops_direction_indep.py
python build_snapshot_matrix_from_vtu.py
python compute_svd.py
python project_to_reduced_ops.py
```

The scripts use NumPy and SciPy. Reading the OpenSn files requires VTK's
Python bindings, and the diagnostic plots require Matplotlib. Large inputs and
generated products are kept below `2D/run/`, outside Git.

## `compute_mass_matrix_from_vtu.py`

**Inputs.** The steady OpenSn VTU file
`2D/run/opensn/aflux_3newss_1000_0.vtu`. The script reads point coordinates,
triangle connectivity, and the cell-data material identifier `Block`.

**Operation.** For each triangular cell $K$, it computes the area and the
consistent P1 finite-element mass matrix

\[
M_K=\frac{|K|}{12}
\begin{bmatrix}
2&1&1\\
1&2&1\\
1&1&2
\end{bmatrix}.
\]

The VTU point numbering is used directly. This preserves OpenSn's duplicated,
cell-local discontinuous-Galerkin degrees of freedom instead of merging them
into a continuous mesh. Element contributions are collected by material and
assembled as sparse COO/CSC matrices.

**Outputs.** `2D/run/preparation/dg_mass_out/` receives one
`mass_mat_<material>.npz` per material, the global mass matrix, cell
connectivity, cell areas, material IDs, and compact element-mass data. The
original script also contains optional sparsity plotting.

## `build_full_order_ops_direction_indep.py`

**Inputs.** The material mass matrices from `dg_mass_out/` and the material
tables `sigt.txt`, `sigs.txt`, `sigf.txt`, `ivel.txt`, and `qext.txt`.

**Operation.** If $M_m$ is the mass matrix restricted to material $m$, the
script forms sparse direction-independent spatial operators such as

\[
M_t=\sum_m \sigma_{t,m}M_m,\qquad
M_v=\sum_m v_m^{-1}M_m,\qquad
M_s=\sum_m \sigma_{s,m}M_m,
\]

together with $M_1=\sum_m M_m$, the production matrix $M_f$, and the
external-source matrix $M_q$. The forcing vector is evaluated as
`fq = Mq @ ones`.

**Outputs.** Sparse `ops_Mt.npz`, `ops_M1.npz`, `ops_Mv.npz`, `ops_Ms.npz`,
`ops_Mf.npz`, and `ops_Mq.npz`, plus `ops_meta.npz`, are written to
`2D/run/preparation/full_order_out/`. The metadata archive records dimensions,
material coefficients, forcing, and matrix paths.

## `build_snapshot_matrix_from_vtu.py`

**Inputs.** The 1001 transient angular-flux VTUs in
`2D/run/opensn/transient/` and the steady VTU in `2D/run/opensn/`. Point-data
arrays whose names begin with `psi_g000_a` are selected.

**Operation.** Files and angular-field names are naturally sorted. For every
time file, the direction fields are flattened and concatenated in their VTU
ordering, and the correspondingly stacked steady field is subtracted. The
script checks that field names and lengths remain consistent across files and
that 1001 snapshots were found.

A temporary NumPy memmap holds the large matrix while columns are assembled,
so a second full in-memory work array is not required during construction.
The final compressed NPZ must still be read or written as a complete array,
which the source comments note explicitly.

**Outputs.** The centered snapshot matrix and its file/field metadata are
stored under key `X` in
`2D/run/preparation/snapshots/psi_matrix_centered_fp64.npz`.

## `compute_svd.py`

**Inputs.** The centered snapshot matrix $D$ and the spatial matrix $M_1$
identified by `full_order_out/ops_meta.npz`.

**Operation.** This is the original legacy mass-weighted Gram-matrix POD. For
direction blocks $D_d$, it accumulates

\[
C=\sum_d D_d^T M_1 D_d
  =D^T(I_{N_{\mathrm{dir}}}\otimes M_1)D
\]

in `float64`, symmetrizes $C$, and solves its symmetric eigensystem. The
singular values are
\(\sigma_i=\sqrt{\max(\lambda_i,0)}\), and the first 120 physical modes are
formed as $U=DV\operatorname{diag}(1/\sigma_i)$.

The Gram accumulation is performed one angular direction at a time. Basis
construction is row-blocked, and the large `float32` basis is written through
a NumPy `.npy` memmap. The mass-weighted coefficients are likewise accumulated
by direction into a `float64` memmap. These choices limit unnecessary full-
array copies after the compressed snapshot archive has been loaded.

**Outputs.** `2D/run/preparation/svd_out/` receives the rank-120 basis,
coefficients, right singular vectors, eigenvalues, retained singular values,
CSV/PNG diagnostics, and metadata. `singular_values.pdf` plots the complete
computed singular-value spectrum on a logarithmic y-axis and marks $K=120$.
The `singular_values.pdf` preserved here is the singular-value spectrum
generated by `compute_svd.py` with $K=120$.

This legacy POD is preserved for its own value; it is not presented as the
distinguished-first-snapshot POD used by the final paper workflow.

## `project_to_reduced_ops.py`

**Inputs.** The full-order operator metadata, `w.txt`, and the basis named by
`U_PATH`. It points to the rank-120 basis `U_modes_K120.npy` produced by the
preceding `compute_svd.py` step.

**Operation.** For block-diagonal angular operators, the script evaluates

\[
U^T(I\otimes M)U=\sum_d U_d^T M U_d
\]

without constructing the full Kronecker matrix. For scattering and production
operators whose angular block rows are identical, it forms

\[
A=\sum_d U_d,\qquad B=\sum_d w_dU_d,\qquad G=A^TMB,
\]

which uses the rank-one angular structure directly. The forcing is also
projected direction by direction.

**Outputs.** `Gt`, `G1`, `Gv`, `Gs`, `Gf`, `f_red`, weights, material data, and
path metadata are saved in
`2D/run/preparation/quad_forms_out/reduced_ops.npz`.

## Material and quadrature data

Each material table has two columns: material ID and coefficient.

| File | Quantity and use |
|---|---|
| `sigt.txt` | Total macroscopic cross section \(\sigma_t\); weights the material masses used to construct `Mt`. |
| `sigs.txt` | Scattering macroscopic cross section \(\sigma_s\); weights the material masses used to construct `Ms`. |
| `sigf.txt` | Effective production coefficient \(\nu\sigma_f\) used directly by the code; weights the material masses used to construct `Mf`. |
| `ivel.txt` | Inverse particle velocity \(1/v\); weights the material masses used to construct `Mv`. |
| `qext.txt` | External volumetric-source coefficient; weights the material masses used to construct `Mq` and hence `fq`. |
| `w.txt` | The 32 normalized angular quadrature weights, in the same direction order as the VTU angular fields; used by `project_to_reduced_ops.py` for angular scattering/production coupling. |

## Useful implementation patterns retained here

- VTK readers extract angular point fields, triangular connectivity, and cell
  material IDs directly from OpenSn output.
- The discontinuous cell-local point ordering is carried from the VTUs through
  mass assembly and snapshot stacking.
- Sparse material matrices provide a compact affine decomposition of the
  direction-independent operators.
- Natural sorting keeps numbered VTUs and angular fields in physical order.
- Temporary and output memmaps, row blocks, angular-direction blocks, and
  sparse-matrix/dense-matrix products reduce avoidable memory duplication.
- Reduced angular operators are evaluated from their block structure instead
  of forming large Kronecker matrices explicitly.
