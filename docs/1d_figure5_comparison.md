# Phase 7 sigmoid Figure 5 comparison

This report compares run `phase7-20260802T015535Z` with the qualitative Figure
5 trends described by the manuscript. It is a regenerated sigmoid-benchmark
study, not an exact historical reproduction. The current corrected derivative
and indexing behavior, the author-approved aggregate metric, the author-
approved solve-only timing boundary, current Apple M2 Max hardware, and the
absence of original result provenance all limit direct numerical comparison.

## Validated series

`N_q` is `[1, 2, 4, 8, 16, 32, 64, 128]`. Errors use
`relative_space_time_l2_error_v1`; speed-ups use
`rom_solve_ivp_only_v1` and the exact FOM integration time
`662.6050200000172` seconds.

### Projected operators

| Series | Relative errors over increasing `N_q` |
|---|---|
| Linear, `N_r=32` | 2.55576e-2 (one result reused at all points) |
| Elementwise | 2.52234e-2, 2.54200e-2, 2.54305e-2, 2.52971e-2, 2.53215e-2, 2.53258e-2, 2.53239e-2, 2.53239e-2 |
| Tensorial | 1.85482e-2, 1.64960e-2, 1.47771e-2, 1.28637e-2, 9.86238e-3, 9.05585e-3, 9.14874e-3, 9.17894e-3 |
| Enlarged linear, `32+N_q` | 2.46982e-2, 2.39304e-2, 2.24452e-2, 2.00630e-2, 1.54650e-2, 9.59151e-3, 3.58672e-3, 4.69571e-4 |
| Best M-projection, `32+N_q` | 1.51138e-2, 1.45556e-2, 1.35198e-2, 1.16767e-2, 8.72156e-3, 4.95939e-3, 1.63142e-3, 1.77416e-4 |

| Integrated series | Online speed-ups over increasing `N_q` |
|---|---|
| Linear, `N_r=32` | 555.66 (one result reused at all points) |
| Elementwise | 465.40, 448.88, 445.40, 453.38, 450.73, 444.13, 444.71, 441.82 |
| Tensorial | 231.71, 242.98, 251.99, 273.59, 318.01, 315.00, 301.34, 282.95 |
| Enlarged linear, `32+N_q` | 500.70, 511.95, 492.18, 442.40, 381.92, 296.20, 79.60, 65.67 |

### Inferred operators

| Series | Relative errors over increasing `N_q` |
|---|---|
| Linear, `N_r=32` | 2.55576e-2 (one result reused at all points) |
| Elementwise | 2.41460e-2, 2.41498e-2, 2.40907e-2, 2.44549e-2, 2.51654e-2, 2.51943e-2, 2.51895e-2, 2.51900e-2 |
| Tensorial | 1.53452e-2, 1.48036e-2, 1.38165e-2, 1.22143e-2, 9.66137e-3, 8.04313e-3, 8.27117e-3, 8.27333e-3 |
| Enlarged linear, `32+N_q` | 2.46982e-2, 2.39304e-2, 2.24452e-2, 2.00630e-2, 1.54650e-2, 9.59151e-3, 3.58672e-3, 4.69581e-4 |
| Best M-projection, `32+N_q` | 1.51138e-2, 1.45556e-2, 1.35198e-2, 1.16767e-2, 8.72156e-3, 4.95939e-3, 1.63142e-3, 1.77416e-4 |

| Integrated series | Online speed-ups over increasing `N_q` |
|---|---|
| Linear, `N_r=32` | 537.08 (one result reused at all points) |
| Elementwise | 440.48, 435.24, 462.13, 458.16, 455.48, 453.26, 442.10, 440.86 |
| Tensorial | 431.79, 434.74, 434.36, 437.69, 446.45, 434.21, 386.55, 407.50 |
| Enlarged linear, `32+N_q` | 521.43, 504.64, 493.62, 442.66, 378.26, 295.97, 84.50, 61.95 |

The best-projection benchmark is shared between panels and has no online time
or speed-up.

## Solver, inference, and stage diagnostics

All 50 integrations succeeded and reached exactly `t=10`. Across series,
`nfev` ranged from 44,729 to 203,889, `njev` from 2 to 16, and `nlu` from 12
to 132. The bundle contains the exact values for every case.

All 16 nonlinear inferred cases converged from zero linear/quadratic operator
initialization. Elementwise cases used 10,279 iterations with final convergence
measures from `9.99925e-7` to `9.99995e-7`; tensorial cases used 8,677–8,690
iterations with measures from `9.99896e-7` to `9.99975e-7`. Exact residuals
and operator norms are retained per case. Linear inference is recorded as a
closed-form solve with no alternating minimization.

| Panel/series | Lifting (s) | Projection (s) | Inference (s) | Initial fit (s) | Online (s) | Reconstruction (s) | Metric (s) | Total workflow (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Projected fixed linear | 0.000001 | 0.0828 | 0 | 0.000003 | 1.1925 | 0.1841 | 0.2551 | 2.9950 |
| Projected elementwise | 0.0043–0.0171 | 0.0481–0.0622 | 0 | 0.0695–0.1180 | 1.4237–1.4997 | 0.2950–0.3216 | 0.2383–0.2542 | 2.4831–2.5893 |
| Projected tensorial | 0.0918–0.1207 | 0.0533–0.0691 | 0 | 0.0934–0.5567 | 2.0836–2.8597 | 0.7431–0.8037 | 0.2540–0.2851 | 3.9352–4.5862 |
| Projected enlarged linear | 0.000001–0.000003 | 0.0504–0.1665 | 0 | 0.000002–0.000009 | 1.2943–10.0902 | 0.1727–0.3587 | 0.1826–0.2606 | 2.1580–11.1826 |
| Inferred fixed linear | 0.000002 | 0.0639 | 0.0170 | 0.000003 | 1.2337 | 0.1515 | 0.1809 | 2.0376 |
| Inferred elementwise | 0.0041–0.0136 | 0.0505–0.0667 | 0.2902–0.3155 | 0.0640–0.1104 | 1.4338–1.5224 | 0.2837–0.3124 | 0.2332–0.2643 | 2.8300–2.9138 |
| Inferred tensorial | 0.0897–0.1184 | 0.0585–0.0951 | 2.8398–3.4245 | 0.0890–0.4875 | 1.4842–1.7142 | 0.7264–0.7871 | 0.2540–0.3024 | 6.0234–7.1690 |
| Inferred enlarged linear | 0.000001–0.000006 | 0.0481–0.1602 | 0.0046–0.0374 | 0.000001–0.000013 | 1.2707–10.6957 | 0.1822–0.2560 | 0.1934–0.2638 | 2.1540–11.7111 |

These timings are machine-specific and are not golden assertions.

## Qualitative comparison

| Trend | Classification | Assessment |
|---|---|---|
| Fixed linear error is independent of `N_q` | Supported | Each panel references one rank-32 solve, so its error and speed-up are exactly constant across plotting points. |
| Elementwise improvement with increasing `N_q` | Not supported | Projected values remain near `2.53e-2`; inferred values are best at small `N_q` and rise toward `2.52e-2`. |
| Tensorial improvement at small-to-moderate `N_q` | Supported | Error decreases through `N_q=32` in both panels. |
| Tensorial error saturation | Supported | The minimum occurs at `N_q=32`; values at 64 and 128 are slightly higher and effectively plateau. |
| Enlarged-linear continued improvement | Supported | Error decreases monotonically from about `2.47e-2` to `4.70e-4`. |
| Best-projection decay | Supported | The shared benchmark decreases monotonically from `1.51e-2` to `1.77e-4` and remains below the corresponding enlarged-linear errors. |
| Projected versus inferred error trends | Supported, numerically different | Qualitative shapes agree. Inferred tensorial errors are lower here; elementwise differences are modest. Exact historical values are unavailable. |
| Online speed-up behavior | Not directly comparable | Values use the approved solve-only boundary and current hardware, neither verified as the historical manuscript environment. |
| Manuscript claim of mild sensitivity of nonlinear online cost to `N_q` | Supported | The regenerated sigmoid results support the manuscript's claim that nonlinear online cost is only mildly sensitive to increasing `N_q` at fixed latent dimension. They do not support a claim of strong nonlinear cost growth with `N_q`. Elementwise online times remain about 1.4–1.5 s, inferred tensorial times remain about 1.5–1.7 s, and projected tensorial times are non-monotone. Offline lifting/inference costs are intentionally excluded. |
| Enlarged-linear cost growth | Supported | Online time rises to 10.09–10.70 s at dimension 160 and speed-up falls to about 62–66. |

## Artifact source

The complete scalar data, per-case solver diagnostics, inference diagnostics,
regularization coefficients and applied ridges, timing components, case paths,
and checksums are in
`results/1d/publication/figure_data/figure5/phase7-20260802T015535Z`.
`complete_publication_reproduction` remains `false`.
