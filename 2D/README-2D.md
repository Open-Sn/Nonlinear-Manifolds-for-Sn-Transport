# 2D reproduction workflow

The two-dimensional reproduction is organized as three sequential stages:

```text
01_run_opensn
      ↓
OpenSn angular-flux snapshots and steady state
      ↓
02_prepare_for_rom
      ↓
mass matrices, full-order operators, centered X1001 snapshots, and center vector
      ↓
03_execute_rom
```

## Stage 1: OpenSn full-order calculation

[`01_run_opensn/`](01_run_opensn/) contains the OpenSn inputs, mesh, and
cross-section file used to generate the angular-flux data. See the
[Stage 1 instructions](01_run_opensn/README.md).

## Stage 2: Preparation for ROM

[`02_prepare_for_rom/`](02_prepare_for_rom/) contains the actual
paper-reproduction preprocessing path. It constructs the mass matrices,
full-order spatial operators, centered snapshot matrix, and center vector
needed by Stage 3. POD is computed by the Stage 3 paper ROM driver, not by
Stage 2.

## Preserved legacy preprocessing

[`02_prepare_for_rom_legacy/`](02_prepare_for_rom_legacy/) preserves Jean
Ragusa's early 2D preprocessing implementation for reference and posterity.
It contains useful preprocessing techniques but is not claimed to be the
exact code path used for every final paper result. See the
[legacy instructions](02_prepare_for_rom_legacy/README.md).

Large VTU files and generated scientific products are intentionally excluded
from Git. The archived OpenSn data are available from
[Zenodo](https://doi.org/10.5281/zenodo.21762243) as
`3newh_aflx.tar` and `aflux_3newss_1000_0.vtu`; see the
[Stage 1 instructions](01_run_opensn/README.md#obtaining-the-opensn-outputs).

## Stage 3: ROM execution

[`03_execute_rom/`](03_execute_rom/) computes the mass-weighted POD, builds
the nonlinear-manifold ROMs, infers the reduced streaming operators, and
generates the 2-D paper figures. See the
[Stage 3 instructions](03_execute_rom/README.md).

## Provenance

The [legacy source manifest](02_prepare_for_rom_legacy/SOURCE_MANIFEST.md)
belongs to the archived implementation; it is not an inventory or checksum
specification for the active Stage 2 workflow.
