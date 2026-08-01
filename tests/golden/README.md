# Independent tiny 1-D golden reference

This directory contains verification data for a six-cell, 48-unknown transport
problem. It is deliberately small and is **not** a publication-scale result or
an authority for the published production calculation.

The tiny transient starts from zero angular flux. It does not represent, sample,
or replace the protected sigmoid initial condition used by the production FOM.

## Files and authority

- `tiny_1d_reference.npz` contains compact numerical arrays. Tabulated
  quadrature, the fixed mesh, material identifiers, and time grid have
  `analytic` authority. Operators, steady and transient states, and POD
  quantities have `independent_numerical` authority because they are produced
  by a separately implemented dense numerical path.
- `tiny_1d_manifest.json` records the configuration, array shapes and dtypes,
  authority of every array, generator and package versions, mathematical
  invariants, comparison tolerances, and the canonical content checksum.
- `../reference_generators/generate_tiny_1d_reference.py` is the standalone
  generator. It imports only the standard library, NumPy, and SciPy; it does not
  import project assembly, driver, ROM, or pytest-fixture code.

No `regression_only` values are included.

## Array contents

| Array | Meaning |
|---|---|
| `mu`, `weights` | Tabulated GL4 ordinates and weights normalized to sum to one |
| `cell_edges`, `cell_material_ids` | Six-cell mesh and zero-based material map |
| `mass_matrix` | Direction-major phase-space DG mass matrix, M |
| `streaming_matrix` | Upwind DG streaming matrix, G |
| `total_interaction_matrix` | Total-interaction matrix, A |
| `scattering_matrix` | Isotropic scattering matrix, B |
| `system_matrix` | F = G + A - B |
| `boundary_inflow_matrix`, `boundary_source` | Physical inflow map and unit left-inflow source b |
| `steady_state` | Dense solution of F psi_inf = b |
| `time`, `transient_state` | Six output times and the augmented-matrix-exponential transient |
| `pod_eigenvalues` | Eigenvalues of C = S.T M S, with no 1/Ns factor |
| `pod_retained_energy`, `pod_unresolved_energy` | Cumulative retained and unresolved correlation-energy fractions |
| `pod_projector_rank3` | Sign-invariant M-orthogonal projector V3 V3.T M |
| `pod_projection_error_rank3` | Absolute M-norm best-projection error per centered snapshot |

The state ordering is direction-major. Within a direction, cells run from left
to right; within each cell, the left endpoint degree of freedom precedes the
right endpoint degree of freedom.

## Independent construction

The generator integrates the two endpoint basis functions analytically to form
each linear-DG mass block. It independently applies the positive- and
negative-direction upwind weak forms, including neighbor and physical inflow
terms, then builds dense direction-major M, G, A, and B matrices.

The steady state uses `scipy.linalg.solve`. The transient does not use Radau or
the production right-hand side: it exponentiates the augmented homogeneous
system containing `-M^{-1}F` and `M^{-1}b` with `scipy.linalg.expm`. POD data is
formed independently from the symmetric correlation eigenproblem
`S.T @ M @ S` using `scipy.linalg.eigh`. Only non-negligible positive
eigenvalues are used to construct the rank-three basis, and the stored
projector is invariant to basis-vector signs.

## Regeneration and verification

From the repository root, regenerate the committed files with:

```bash
python tests/reference_generators/generate_tiny_1d_reference.py
```

Verify regenerated in-memory content without rewriting files with:

```bash
python tests/reference_generators/generate_tiny_1d_reference.py --check
```

To write a disposable copy elsewhere:

```bash
python tests/reference_generators/generate_tiny_1d_reference.py --output-dir /tmp/tiny-golden
```

The check compares names, shapes, dtypes, and numerical values using the
manifest tolerances. It also recomputes the committed artifact's canonical
content checksum. The checksum is SHA-256 over arrays in sorted-name order; for
each array, length-prefixed name, NumPy `dtype.str`, compact-JSON shape, and
contiguous C-order bytes are hashed. NPZ ZIP bytes are intentionally excluded.

## Tolerance and platform policy

- Tabulated quadrature and direct algebraic assembly use absolute tolerances of
  2e-15 and 1e-13, respectively.
- Steady-state comparison uses `rtol=1e-10`, `atol=1e-12`.
- Production-side Radau versus matrix exponential uses `rtol=1e-8`,
  `atol=1e-10`, consistent with the tiny Radau solve tolerances.
- POD spectra, projectors, and projection errors use conditioning-aware
  tolerances recorded in the manifest, no looser than `rtol=1e-8` and
  `atol=1e-10` for the projector.

Directly assembled values should be effectively platform invariant. Dense
linear solves, matrix exponentials, and symmetric eigensolvers can vary by a
few final digits across BLAS/LAPACK implementations. Numerical agreement, not
bitwise NPZ identity or raw POD-vector signs, is therefore authoritative.
