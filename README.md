# Nonlinear-Manifold Reduced-Order Models for Neutron Transport

This repository accompanies the paper *Learning Low-Rank Neutron Transport Dynamics on Linear Subspaces and Nonlinear Manifolds: A Semi-Intrusive Operator-Inferred Approach* (in revision). It contains reproducibility implementations for one-dimensional and two-dimensional neutron-transport benchmarks using related linear and nonlinear-manifold reduced-order modeling methods.

The two cases are intentionally maintained in separate directories so that each numerical study can be reproduced independently.

## Repository organization

| Directory | Description |
|---|---|
| [`1D/`](1D/) | 1-D transport benchmark, ROM implementation, and Figures 1–5 reproduction workflow. |
| [`2D/`](2D/) | Three-stage 2-D workflow: OpenSn full-order calculation, preprocessing/POD, and reserved ROM reproduction. |

## 1-D case

The 1-D case contains the complete transport benchmark, snapshot generator, projected/intrusive and inferred/semi-intrusive ROM studies, and Figure 1–5 generation.

See the [1-D reproducibility instructions](1D/README-1D.md) for the benchmark definition, snapshot generation, ROM configuration, execution workflow, and figure reproduction.

## 2-D case

The recovered 2-D full-order and POD reproduction is organized under `2D/`:

| Stage | Purpose |
|---|---|
| `01_run_opensn/` | Reproduce the transient OpenSn full-order calculation and steady reference. |
| `02_prepare_for_rom/` | Reconstruct mass/operator data, build the centered angular snapshot matrix, and compute the mass-weighted POD. |
| `03_execute_rom/` | ROM reproduction and remaining 2-D paper results; to be populated next. |

See the [2-D reproducibility instructions](2D/README-2D.md) for the staged
workflow and current scope.

## License

See [`LICENSE`](LICENSE).
