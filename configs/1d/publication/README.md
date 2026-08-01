# Sigmoid-based publication experiment metadata

Every publication-oriented 1-D experiment in this repository derives from
`configs/1d/legacy_production.json`, whose canonical SHA-256 checksum is
`cc442174134332f4b722cfa65ef179e1abc350c3e27e342a8bfeb184aa1b2759`.
The catalog does not duplicate or replace that benchmark configuration.

## Benchmark policy

The authoritative repository benchmark intentionally places the localized
sigmoid

```text
1 - 1 / (1 + exp(-100 * (x - 0.1)))
```

in the final positive angular block. Its amplitude is one, its transition is
at `x=0.1`, and its steepness is `100`. The manuscript states zero initial
angular flux. The repository deliberately retains the sigmoid, likely to
smooth startup and reduce the sharp incompatibility between zero flux and a
suddenly imposed boundary inflow. This is a small, intentional implementation
deviation—not a defect, a pending code fix, or an unresolved author question.
No zero-initial-condition production configuration is maintained.

Every resolved case and future result manifest records
`"benchmark_variant": "legacy_sigmoid"` and the structured manuscript
deviation. This catalog does not claim bit-for-bit or exact reproduction of
the manuscript figures.

## Encoded values and sources

`experiments.json` records values supplied for this Phase 4 publication
workflow and distinguishes them from preserved repository provenance:

The resolved catalog contains 57 cases: 1 for Figure 1, 3 for Figure 2,
3 for Figure 3, 48 for Figure 4, and 2 aggregate Figure 5 studies.

| Study | Encoded values | Source/status |
|---|---|---|
| Figure 1 | Training `[0,7.5]`, steady-state centering, `N_r=16`, `N_q=548`, total 564 | Figure 1 specification; fully specified |
| Figure 2 | Projected linear, elementwise, tensorial; `N_r=16`, nonlinear `N_q=548` | Figure 2 specification; fully specified |
| Figure 2 lifting | Elementwise `gamma=6.4e-6`; tensorial `gamma=2.5e-6` | Reported figure values |
| Figure 3 | Inferred linear, elementwise, tensorial; `N_r=16`, nonlinear `N_q=548` | Figure 3 specification; fully specified |
| Figure 3 inference | `lambda_L=0`; elementwise `lambda_Q=2.56e-5`; tensorial `lambda_Q=1.6e-6`; tolerance `1e-6`; maximum 100,000 iterations | Reported values plus existing production safeguard |
| Figure 3 iterations | Elementwise 23,367; tensorial 868 | Manuscript metadata only; never stopping targets or regression assertions |
| Figure 4 ranks | `N_r=[8,16,24,32,40,48,56,64]`; nonlinear `N_q=564-N_r`; linear dimension `N_r` with `N_q` not applicable | Reported study design |
| Figure 4 series | Linear, elementwise, and tensorial, each with projected and inferred operators | Six series; 48 cases total |
| Figure 4 search ranges | `gamma` from approximately `7e-10` to `5e-5`; `lambda_Q` from approximately `6e-9` to `2e-4`; `lambda_L=0` | Candidate ranges only |
| Figure 4 nonlinear selections | Per-rank/per-lifting selected `gamma` and inferred `lambda_Q` | Not available; requires author input |
| Figure 4 linear status | Model dimension `N_r`; inferred `lambda_L=0`; no nonlinear regularization or alternating-minimization controls | Model parameters complete; aggregate metric and timing classification unresolved |
| Figure 5 ranks | `N_r=32`; `N_q=[1,2,4,8,16,32,64,128]` | Reported study design |
| Figure 5 tensorial | `gamma=2.5e-8`; inferred `lambda_L=0`, `lambda_Q=4e-7` | Reported constants |
| Figure 5 elementwise | `gamma=8e-7`; inferred `lambda_L=0`, `lambda_Q=8e-7` | Reported constants |

Projected cases never carry `lambda_L` or `lambda_Q`. Linear cases never
carry nonlinear lifting `gamma` or `lambda_Q`. Figure 5 records fixed-rank
linear, nonlinear, enlarged-linear, and M-orthogonal projection-benchmark
series separately.

The nonlinear constraint `N_r+N_q=564` never applies to linear Figure 4
models. Linear projected cases carry no regularization. Linear inferred cases
carry only `lambda_L=0`; they do not carry nonlinear inference tolerance or
maximum-iteration metadata because their inference solve is closed-form.

The exact time-aggregated relative convergence metric used by Figures 4 and 5
is not established by repository code or supplied provenance. Publication-
comparable Figure 4/5 execution is therefore refused. Exact online timing
boundaries likewise require explicit classification.

## Figure 4 author input

`figure4_selected_parameters.schema.json` defines the future input contract.
It contains no fabricated selections. A valid file must identify the catalog,
selection objective, search-result provenance, author approval, and selected
values for each case. Candidate ranges alone are never used to select a best
value.
