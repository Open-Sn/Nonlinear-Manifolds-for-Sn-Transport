# Nonlinear-Manifold Reduced-Order Models for Neutron Transport

This repository accompanies the paper *Learning Low-Rank Neutron Transport Dynamics on Linear Subspaces and Nonlinear Manifolds: A Semi-Intrusive Operator-Inferred Approach* (in revision). It contains reproducibility implementations for one-dimensional and two-dimensional neutron-transport benchmarks using related linear and nonlinear-manifold reduced-order modeling methods.

The two cases are intentionally maintained in separate directories so that each numerical study can be reproduced independently.

## Repository organization

| Directory | Description |
|---|---|
| [`1D/`](1D/) | 1-D transport benchmark, ROM implementation, and Figures 1–5 reproduction workflow. |
| [`2D/`](2D/) | Reserved for the separate 2-D transport benchmark and ROM reproduction workflow, which will be added later. |

## 1-D case

The 1-D case contains the complete transport benchmark, snapshot generator, projected/intrusive and inferred/semi-intrusive ROM studies, and Figure 1–5 generation.

See the [1-D reproducibility instructions](1D/README-1D.md) for the benchmark definition, snapshot generation, ROM configuration, execution workflow, and figure reproduction.

## 2-D case

The separate 2-D implementation and results will live under `2D/`. Detailed reproducibility instructions will be added as `2D/README-2D.md` when that workflow is finalized.

## License

See [`LICENSE`](LICENSE).
