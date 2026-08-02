# Regenerated sigmoid Figure 4 comparison

Phase 8 run `phase8-20260802T130106Z` completed all 48 Figure 4 cases. The
validated bundle is
`results/1d/publication/figure_data/figure4/phase8-20260802T130106Z` with data
checksum `6920cca2ef021abe1e8a553480770781771e0e1991b319646044b164073055f3`.
It is labeled `regenerated_sigmoid_benchmark`; nonlinear selections have
provenance `regenerated_sigmoid_search` and are not recovered historical
values.

## Tracked regenerated selections

The reviewed nonlinear coefficients are tracked in
`configs/1d/publication/figure4_selected_parameters.json` with deterministic
content checksum
`afd75f91a16b8dc3b87484752f2b699203d980ea577fa2721124ac6ae9d43d4e`.
The file contains all 32 model/operator/rank selections, the `N_s=7501`
coefficient-to-ridge scaling, projected-gamma reuse for inferred cases, coarse
or refined origins, tie outcomes, and the Phase 8 source checksums. It contains
only portable paths.

The publication catalog now resolves all 48 Figure 4 cases from tracked
configuration: 16 linear cases are directly specified and 32 nonlinear cases
reference this file. A fresh clone can inspect, dry-run, and rerun the
regenerated sigmoid cases with a valid snapshot and shared-offline inputs
without access to the Phase 8 result directory and without repeating the
search. Repeating the documented Phase 8 search remains available as an
independent verification step.

## Trend assessment

| Manuscript trend | Classification | Regenerated evidence |
|---|---|---|
| Error decreases as `N_r` increases | Supported | Every one of the six error series decreases from `N_r=8` to 64. Endpoint reduction factors range from 7.79 for linear to 23.58 for inferred tensorial. |
| Elementwise behavior is close to linear | Supported | Elementwise/linear error ratios remain within 0.977--1.012 for projected and 0.982--1.004 for inferred operators. |
| Tensorial error converges faster | Supported | Tensorial error falls by factors 20.75 (projected) and 23.58 (inferred), versus about 7.8 for linear and elementwise. At `N_r=64`, tensorial error is 19.9% (projected) and 16.2% (inferred) of linear error. |
| Projected and inferred trends agree | Supported | Linear errors agree to 0.000025% relative. Mean projected/inferred differences are 1.02% for elementwise and 14.0% for tensorial; all retain the same ordering and convergence trend. |
| Online speed-up declines with `N_r` | Supported | Endpoint speed-up declines are 6.13--7.28-fold for linear/elementwise and 22.25--30.22-fold for tensorial. Projected tensorial has a small non-monotone rebound from 89.9 at `N_r=48` to 94.1 at 56, but the overall decline is clear. |
| Tensorial has favorable speed-up at small rank | Supported | At `N_r=8`, tensorial speed-up is 2048 projected and 2021 inferred, 11.9% and 11.1% above the corresponding linear values. |
| Larger ranks reduce the tensorial speed-up advantage | Supported | By `N_r=64`, tensorial speed-up is 30.8% of projected linear and 22.6% of inferred linear speed-up. |
| Exact quantitative agreement with manuscript curves | Not directly comparable | No archived numerical manuscript curve arrays or historical selected regularization values were recovered, so pointwise manuscript-minus-regenerated differences cannot be computed. |

## Quantitative regenerated endpoints

| Operators | Model | Error `N_r=8` | Error `N_r=64` | Speed-up `N_r=8` | Speed-up `N_r=64` |
|---|---|---:|---:|---:|---:|
| Projected | Linear | 0.0746931 | 0.00959151 | 1830.0 | 298.6 |
| Projected | Elementwise | 0.0755710 | 0.00951780 | 1670.2 | 238.0 |
| Projected | Tensorial | 0.0396076 | 0.00190875 | 2048.0 | 92.0 |
| Inferred | Linear | 0.0746931 | 0.00959151 | 1819.4 | 296.2 |
| Inferred | Elementwise | 0.0749769 | 0.00952314 | 1697.0 | 233.1 |
| Inferred | Tensorial | 0.0365385 | 0.00154978 | 2020.7 | 66.9 |

## Interpretation limits

The study uses the author-confirmed sigmoid configuration, corrected current
derivative and indexing behavior, the author-approved full-interval aggregate
metric, and solve-only online timing. Nonlinear regularization was selected by
the new checksummed search. Timings are measurements on the current machine.
These differences in benchmark interpretation, implementation behavior,
selection provenance, and hardware mean the plot supports qualitative trend
assessment but is not an exact reproduction of the manuscript calculation.
The tracked configuration improves reproducibility of this regenerated study;
it does not establish or recover the historical manuscript parameters.
