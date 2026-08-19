# Nonlinear-Manifold Reduced-Order Models for Neutron Transport

This repository accompanies the paper *Learning Low-Rank Neutron Transport Dynamics on Linear Subspaces and Nonlinear Manifolds: A Semi-Intrusive Operator-Inferred Approach* (in revision). It contains reproducibility implementations for one-dimensional and two-dimensional neutron-transport benchmarks using related linear and nonlinear-manifold reduced-order modeling methods.

The two cases are intentionally maintained in separate directories so that each numerical study can be reproduced independently.

## Repository organization

| Directory | Description |
|---|---|
| [`1D/`](1D/) | 1-D transport benchmark, ROM implementation, and Figures 1–5 reproduction workflow. |
| [`2D/`](2D/) | Three-stage 2-D workflow: OpenSn full-order calculation, ROM preparation, and paper ROM execution. |

## 1-D case

The 1-D case contains the complete transport benchmark, snapshot generator, projected/intrusive and inferred/semi-intrusive ROM studies, and Figure 1–5 generation.

See the [1-D reproducibility instructions](1D/README-1D.md) for the benchmark definition, snapshot generation, ROM configuration, execution workflow, and figure reproduction.

## 2-D case

The 2-D paper-reproduction path is organized under `2D/`:

| Stage | Purpose |
|---|---|
| `01_run_opensn/` | Full-order transient OpenSn problem and steady reference. |
| `02_prepare_for_rom/` | DG mass matrices, full-order spatial operators, centered angular-flux snapshots, and center vector. |
| `03_execute_rom/` | Mass-weighted POD, nonlinear-manifold ROMs, operator inference, and paper figures. |

`2D/02_prepare_for_rom_legacy/` preserves archival/reference preprocessing
code and is not part of the active reproduction chain.

See the [2-D reproducibility instructions](2D/README-2D.md) for the staged
workflow. The large 2-D OpenSn outputs are stored on
[Zenodo](https://doi.org/10.5281/zenodo.21762243) rather than in Git.

## License

See [`LICENSE`](LICENSE).
