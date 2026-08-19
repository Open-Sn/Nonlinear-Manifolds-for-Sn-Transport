# Stage 1: Run the 2-D OpenSn full-order problem

[Back to the 2-D workflow](../README-2D.md)

This calculation uses [OpenSn](https://open-sn.github.io/opensn/), the open-source discrete-ordinates transport code. The preserved inputs were used successfully with OpenSn as it existed in **February 2026**; the exact historical commit/version was not retained.

These inputs are not part of the OpenSn automated testing suite because the transient calculation is comparatively expensive. If future OpenSn interface changes make the preserved inputs incompatible with the current code, consult the current OpenSn documentation or the [OpenSn Discussions](https://github.com/Open-Sn/opensn/discussions).

## Purpose

This stage preserves the inputs for the transient two-dimensional OpenSn full-order calculation and the long-time calculation used as the steady reference.

The steady angular flux is required because the ROM study is performed on
**centered snapshots**: Stage 2 subtracts the steady solution from every
transient angular-flux snapshot. Stage 3 computes the POD from that centered
matrix.

## Files

| File | Role |
|---|---|
| `td7sq_final.py` | Transient OpenSn solve and scalar/angular VTU export. |
| `td7sq_final_ss.py` | Long backward-Euler march and final steady-reference export. |
| `square.msh` | Gmsh mesh with 3,741 original nodes and 7,280 triangles. |
| `xs_fis_vel_src2.xs` | One-group fissile cross section, production data, and velocity. |

## Transport benchmark

The problem has one energy group and uses

```text
GLCProductQuadrature2DXY(
    n_polar=2,
    n_azimuthal=32,
    scattering_order=0
)
```

The retained OpenSn output contains 32 angular fields. The spatial discretization has 21,840 cell-local piecewise-linear discontinuous unknowns, so the full angular state contains

```text
32 × 21,840 = 698,880
```

unknowns. This state dimension is also confirmed by the centered snapshot
matrix consumed by Stage 3.

Material blocks are mapped as follows:

- fissile: 1, 2, 4, 7, 10, and 11 (`sigma_t=1`, `sigma_f=0.98`, `nu=2`, `v=5`);
- absorbing: 3, 5, 6, 8, and 9 (`sigma_t=1`, no scattering);
- scattering: 12 (`sigma_t=1`, scattering ratio 0.75, `v=5`);
- void background: 13 (`sigma_t=0`, `v=5`).

The configured volumetric source has zero strength and is disabled in the problem definition. On `xmin`, OpenSn angle ID 31 has incoming value 1 and the other directions have value 0. `xmax` and `ymax` have zero isotropic inflow. No `ymin` condition is explicitly recorded; it therefore depended on the default in the historical OpenSn version.

The input also contains an eigenvalue/criticality calculation that was used to verify that the configuration is **subcritical**. This ensures that the source-driven transient relaxes toward the finite steady state used for snapshot centering.

## Transient calculation

`td7sq_final.py` starts from zero angular flux, uses theta `0.5`, nominal `dt=0.005`, and marches to final time `5`. Every accepted step exports scalar and angular fields using the roots

```text
flux_3newh_####
aflux_3newh_####
```

Files 0001--1000 are the maintained physical-time sequence. Floating-point
time accumulation caused one additional near-zero step, so file 1001 is a
near-duplicate of the final state. The historical paper preprocessing retains
all 1001 files.

## Steady-reference calculation

`td7sq_final_ss.py` uses the same mesh, quadrature, materials, source, and boundary definitions. It uses theta `1`, `dt=0.5`, and final time `500`, then exports the final scalar and angular fields with roots

```text
flux_3newss_1000
aflux_3newss_1000
```

The single-rank angular piece needed by Stage 2 is

```text
aflux_3newss_1000_0.vtu
```

Stage 2 subtracts this steady angular state from each transient angular state
to construct the centered snapshot matrix used by Stage 3.

## Expected OpenSn outputs

The transient calculation produces angular pieces named

```text
aflux_3newh_####_0.vtu
```

and scalar descriptors/pieces named

```text
flux_3newh_####.pvtu
flux_3newh_####_0.vtu
```

Each angular VTU contains all 32 direction fields over all 21,840 DG points.

Stage 2 uses **angular flux**. Scalar flux is a derived diagnostic and is not the POD state.

## Historical OpenSn environment

A future rerun with a newer OpenSn version may therefore require interface/import adaptation. Such adaptation should preserve the physical problem and numerical settings documented above.

## Obtaining the OpenSn outputs

### Recommended reproduction path

Download the following files from
[Zenodo](https://doi.org/10.5281/zenodo.21762243):

```text
3newh_aflx.tar
aflux_3newss_1000_0.vtu
```

Place the archive and steady VTU under `2D/run/opensn/`, and extract the
transient angular-flux VTUs under `2D/run/opensn/transient/`. The resulting
layout needed by Stage 2 is:

```text
2D/run/opensn/aflux_3newss_1000_0.vtu
2D/run/opensn/transient/aflux_3newh_0001_0.vtu
...
2D/run/opensn/transient/aflux_3newh_1001_0.vtu
```

These large files remain outside Git.

### Regenerate with OpenSn

The retained input files were used with OpenSn as it existed in February
2026, but the exact historical commit and launch command were not preserved.
A researcher wishing to regenerate the full-order outputs should use the
current OpenSn documentation for the installed version and preserve the
physical and numerical settings recorded above. No exact historical command
is claimed here.

## Output needed by Stage 2

Stage 2 requires:

- angular VTUs `aflux_3newh_0001_0.vtu` through `aflux_3newh_1001_0.vtu`; and
- `aflux_3newss_1000_0.vtu`.

See the [Stage 2 input instructions](../02_prepare_for_rom/README.md#input-data)
for the paths used by the preprocessing scripts.
