# Stage 2: Prepare the 2-D OpenSn data for ROM calculations

[Back to the 2-D workflow](../README-2D.md)

## Purpose

This stage converts the OpenSn full-order angular output into the spatial mass
and material operators, steady-centered snapshot matrix, and mass-weighted POD
products defining the maintained X1000/K80 pre-ROM workflow. These products
provide one set of inputs for the future
[Stage 3 ROM workflow](../03_execute_rom/).

```text
transient angular VTUs + steady angular VTU
                    ↓
DG/material mass matrices
                    ↓
direction-independent FOM matrices
                    ↓
centered angular snapshot matrix X
                    ↓
mass-weighted POD
                    ↓
singular values + K80 modes + right vectors + coefficients
```

No ROM dynamics are constructed or executed in this stage.

## Files

| File | Role |
|---|---|
| `compute_mass_matrix_from_vtu.py` | Builds material-wise and global P1-DG spatial mass matrices. |
| `build_full_order_ops_direction_indep.py` | Builds `Mt`, `M1`, `Mv`, `Ms`, `Mf`, and `Mq`. |
| `build_snapshot_matrix_from_vtu.py` | Natural-sorts angular VTUs, stacks all direction fields, and subtracts the steady state. |
| `compute_svd.py` | Forms the mass-weighted Gram matrix and computes the POD products. |
| `check_pod_mass_orthonormality.py` | Checks `U.T (I_32 kron M1) U` against the identity. |
| `sigt.txt`, `sigs.txt`, `sigf.txt`, `ivel.txt`, `qext.txt` | Thirteen-material coefficient tables used by the full-order operator builder. |
| `w.txt` | Thirty-two normalized angular weights in OpenSn angle order. It supports angular/scalar-flux provenance but is not part of the POD mass. |

## Required Python packages

The preserved scripts require NumPy, SciPy, VTK's Python bindings, and
Matplotlib for their optional plots. The historical validation used NumPy
1.26.4 and SciPy 1.11.4.

Run all commands below with `2D/02_prepare_for_rom/` as the working directory.
The scripts intentionally retain their recovered working-directory-relative
path constants.

## Inputs from Stage 1 or archived data

The default preserved paths expect this data layout, which remains ignored by
Git:

```text
02_prepare_for_rom/
├── aflux_3newss_1000_0.vtu
└── 3newh_aflx/
    ├── aflux_3newh_0001_0.vtu
    ├── ...
    └── aflux_3newh_1000_0.vtu
```

After a local Stage 1 run, **move** the 1,000 maintained transient angular
pieces and the steady angular piece into this layout; moving avoids a second
multi-gigabyte copy. Keep file 1001 outside `3newh_aflx/` as a diagnostic.
When using archived data, extract the same selected files directly into this
layout.

Alternatively, leave the data in any external location and edit only these
existing path constants:

- `VTU_PATH` in `compute_mass_matrix_from_vtu.py`;
- `INPUT_GLOB` and `CENTER_FILE` in `build_snapshot_matrix_from_vtu.py`.

No source-relative path adaptation or scientific-code change is required by
the new repository organization.

## Constructing the DG mass matrices

Run:

```bash
python compute_mass_matrix_from_vtu.py
```

For unattended execution, set the existing plotting option
`PLOT_AFTER_COMPUTE=False`. For a triangle `K` with area `|K|`, the script
uses

\[
M_K=\frac{|K|}{12}
\begin{bmatrix}
2&1&1\\
1&2&1\\
1&1&2
\end{bmatrix}.
\]

Using the VTU connectivity and `Block` material ID, it assembles

\[
M_m=\sum_{K\in m}P_K^T M_K P_K
\]

for materials 1--13, plus the global spatial mass. Expected outputs in
`dg_mass_out/` are `mass_mat_1.npz` through `mass_mat_13.npz`,
`mass_global_mat.npz`, `cell2pt.npy`, `cell_areas.npy`, `cell_material.npy`,
`mass_scale_A_over_12.npy`, and `mass_template_T.npy`.

The verified dimensions are 7,280 triangular cells and 21,840 cell-local
spatial DG unknowns.

## Constructing direction-independent full-order operators

Run, using a noninteractive plotting backend when appropriate:

```bash
MPLBACKEND=Agg python build_full_order_ops_direction_indep.py
```

The script reads the thirteen material mass matrices and material tables and
forms

\[
M_t=\sum_m\sigma_{t,m}M_m,\qquad M_1=\sum_m M_m,
\]

\[
M_v=\sum_m v_m^{-1}M_m,\qquad M_s=\sum_m\sigma_{s,m}M_m,
\]

\[
M_f=\sum_m\nu\sigma_{f,m}M_m,\qquad
M_q=\sum_m q_mM_m.
\]

`sigf.txt` stores the effective coefficient `nu*sigma_f`. The verified data
satisfy `Mv=0.2 M1`; `Mq` and its forcing vector are zero. Outputs are written
under `full_order_out/` as `ops_Mt.npz`, `ops_M1.npz`, `ops_Mv.npz`,
`ops_Ms.npz`, `ops_Mf.npz`, `ops_Mq.npz`, and `ops_meta.npz`.

## Building the centered angular-flux snapshot matrix

Run:

```bash
python build_snapshot_matrix_from_vtu.py
```

The script natural-sorts the transient filenames and the 32 point-data fields
`psi_g000_a000_gs00` through `psi_g000_a031_gs00`. OpenSn's duplicated
cell-local VTU points are used directly as DG unknowns; no continuous-node
merge is performed. The direction-major layout is

```text
X[d*N+p,j] = psi_d(p,t_j) - psi_infinity_d(p).
```

Each column has 698,880 entries. With files 0001--1000, the maintained matrix
has shape `(698880,1000)` and dtype `float64`. The exact preserved script writes
`psi_matrix_centered_fp64.npz` with array key `X` and metadata.

## X1000 versus X1001

Files 0001--1000 represent nominal physical times 0.005--5.000. Floating-point
time accumulation left the 1000th state at approximately 4.999999999999916,
so the OpenSn loop took an additional step of about `8.44e-14`. File 1001 is
therefore a near-duplicate diagnostic and is excluded from the maintained
training matrix.

The canonical maintained contract is:

```text
indices: 0001--1000
shape:   (698880,1000)
dtype:   float64
```

The separately archived convenience product is
`X_0001_1000_centered_fp64.npy`. The historical
`psi_matrix_centered_fp64.npz` with 1,001 columns is X1001 and must not be
substituted for the maintained X1000 workflow.

**Maintained versus historical workflow.** The X1000/K80 workflow documented
in this stage is the maintained preprocessing contract. Exact reproduction of
the published 2-D ROM results follows the recovered historical X1001
conventions and historical POD products. That historical ROM path will be
documented separately in Stage 3.

## Computing the mass-weighted POD

The maintained stable truncation is `K=80`. Set the existing `K` configuration
in `compute_svd.py` to 80 before the expensive run, then execute:

```bash
python compute_svd.py
```

With

\[
\mathbb M=I_{32}\otimes M_1,
\]

the script applies the spatial mass one angular block at a time and computes

\[
C=X^T\mathbb M X,\qquad CV=V\operatorname{diag}(\lambda),
\]

\[
s_i=\sqrt{\max(\lambda_i,0)},\qquad
U=XV\operatorname{diag}(s)^{-1},
\]

with mass-weighted coefficients

\[
A=U^T\mathbb M X.
\]

Expected K80 products under `svd_out/` are:

- `U_modes_K80.npy`: `(698880,80)`, float32 physical POD modes;
- `coeffs_C_K80.npy`: `(80,1000)`, float64 coefficients;
- `singular_values_K80.npy` and its CSV/plot;
- `svd_results_fp32.npz`, including float64 right vectors `V` with shape
  `(1000,80)`, eigenvalues, singular values, and metadata.

## Checking POD mass orthonormality

Run:

```bash
python check_pod_mass_orthonormality.py --rank 80
```

This checks `U.T (I_32 kron M1) U` without constructing reduced transport
operators. The maintained audited K80 basis had maximum absolute mass-Gram
error approximately `2.7003e-5`, captured mass-energy fraction
`0.9999999999997793`, and relative discarded mass energy approximately
`4.7169e-7`.

## Expected outputs

```text
dg_mass_out/                  material/global P1-DG masses and geometry arrays
full_order_out/               Mt, M1, Mv, Ms, Mf, Mq, and metadata
psi_matrix_centered_fp64.npz  regenerated X1000 snapshot matrix
svd_out/                      singular values, K80 modes, V, and coefficients
```

These generated products remain outside Git.

## Handoff to Stage 3

The X1000 snapshot matrix, K80 POD products, spatial mass, and full-order
metadata form the **maintained** pre-ROM data contract.

Stage 3 will distinguish this maintained workflow from exact historical
reproduction of the published 2-D ROM results. The historical path uses the
recovered X1001 and historical basis/operator conventions and will be
documented together with the Stage 3 implementation.