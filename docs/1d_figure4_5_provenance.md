# 1-D Figure 4/5 provenance recovery

This report records the Phase 6 audit of all nine commits reachable through
`git rev-list --all` at `62712f0`, plus current non-large source, configuration,
documentation, test, log, manifest, JSON, CSV, Markdown, and figure-metadata
files. No deleted path appears in `git log --all --name-status`. The existing
large production snapshot, derivative matrices, POD arrays, bases, and full
trajectories were not repeatedly parsed. No Figure 4/5 solve or parameter
search was run.

Subsequent author confirmation establishes that the manuscript's
one-dimensional Figures 1--3 were generated with the localized sigmoid
workflow, despite the manuscript text stating zero initial angular flux. The
repository retains `configs/1d/legacy_production.json` as the authoritative
reproduction configuration for Figures 1--5.

## Aggregate error metric

### Established evidence

The only historical executable error implementation is instantaneous. At
`74bf785:Nonlinear_Manifold_ROM.py:503-509` (unchanged in the corrected and
pre-Phase-3 versions), it computes

\[
e_\infty(t_i)=
\sqrt{\frac{d(t_i)^T M d(t_i)}{\psi_\infty^T M\psi_\infty}},
\qquad d(t_i)=\psi_{\mathrm{ROM}}(t_i)-\psi_{\mathrm{FOM}}(t_i).
\]

The main block then prints `errors.mean()` over every returned time point
(`74bf785:535-540,654-668`). This is an equal-weight arithmetic mean of
instantaneous errors normalized by the steady state. It is executable
historical behavior, but it is not the manuscript-described ratio of a
time-integrated phase-space error to time-integrated reference-solution
energy. There is no `trapz`, trapezoid, Simpson, temporal-weight array, or
other aggregate-error implementation anywhere in reachable history.

Consequently, the publication aggregate metric is **not authoritatively
established** and has not been implemented. The existing instantaneous metric
is unchanged.

### Ranked candidate interpretations (not selected)

| Rank | Candidate | Source support | Unresolved details |
|---:|---|---|---|
| 1 | \(\sqrt{\int_I \|\psi_{FOM}-\psi_{ROM}\|_M^2\,dt / \int_I \|\psi_{FOM}\|_M^2\,dt}\) | Best literal match to “time-integrated phase-space error” divided by reference “energy” | Quadrature, endpoints, interval, centering, and final square-root wording are not executable provenance |
| 2 | \(\int_I \|\psi_{FOM}-\psi_{ROM}\|_M\,dt / \int_I \|\psi_{FOM}\|_M\,dt\) | Plausible if “error” and “energy” were used for unsquared norms | Same quadrature/interval ambiguities; “energy” usually supports squared norms more strongly |
| 3 | \(N_t^{-1}\sum_i e_\infty(t_i)\) | Exact historical executable summary at `74bf785:535-540,654-668` | Contradicts the described transient-reference denominator and ratio-of-integrals form |

Uniform-grid rectangle sums and arithmetic means cancel the common `dt` in a
ratio, but they do not establish endpoint treatment. Trapezoidal and Simpson
weights generally produce different values. The available Figure 2/3 error
histories were therefore not used to choose among these alternatives.

### Author-decision table

| Decision | Source-supported options | Status |
|---|---|---|
| Pointwise numerator | \(\|d(t)\|_M\) or \(\|d(t)\|_M^2\) | Unresolved |
| Denominator field | Full transient FOM is favored by the manuscript description; centered transient and steady state remain possible implementation variants | Unresolved |
| Denominator power | Norm or squared norm/energy | Unresolved |
| Temporal rule | Equal-weight sum/mean, rectangle, trapezoid, Simpson, or another rule | Unresolved |
| Endpoint weights | Equal, half-weight, Simpson pattern, or other | Unresolved |
| Final square root | Applied or not applied | Unresolved |
| Interval | `[0,10]` has the strongest historical support; `[0,7.5]`, `(7.5,10]`, or another interval are not excluded by publication provenance | Unresolved |

## Figure 4 selected parameters

No authoritative selected `gamma` or inferred `lambda_Q` was found. Commit
`35b5989` records only candidate ranges (`gamma` approximately `7e-10` to
`5e-5`, `lambda_Q` approximately `6e-9` to `2e-4`) and reported
`lambda_L=0`; it explicitly says no best values are selected. The original
driver's scalar `N_r=16`, `N_q=364` defaults are a different model dimension
and cannot be copied into the Figure 4 `N_r+N_q=564` study.

`lambda_L=0` is classified `documented_but_unverified`. For inferred cases its
applied Gram ridge is unambiguously zero regardless of scaling. For projected
cases it is reported study metadata but is not applied. All missing `gamma`
and applicable `lambda_Q` values are `not_found`. No filename-derived or stale
comment candidate was found.

The historical selection-run training count is also unresolved. The original
driver selects 7,500 snapshots (`74bf785:161-169`); the corrected/current
inclusive `[0,7.5]` workflow selects 7,501 (`bff6910:266-280`). Neither proves
which count accompanied missing Figure 4 selections. Thus no selected value
can be classified as a paper coefficient or a final pre-scaled ridge, and no
applied `gamma`/`lambda_Q` Gram ridge can be computed.

| N_r | N_q | Model | Operators | gamma | lambda_L | lambda_Q | Coefficient/ridge semantics | Source | Confidence | Execution readiness |
|---:|---:|---|---|---|---|---|---|---|---|---|
| 8 | 556 | elementwise | projected | — | 0 (reported; not applied) | N/A | Selection count and gamma coefficient/ridge semantics unresolved | `35b5989` ranges/zero only; no result | gamma `not_found`; lambda_L `documented_but_unverified` | Blocked: gamma + metric + timing |
| 16 | 548 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 24 | 540 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 32 | 532 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 40 | 524 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 48 | 516 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 56 | 508 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 64 | 500 | elementwise | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 8 | 556 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Selection count and gamma/lambda_Q coefficient/ridge semantics unresolved | `35b5989` ranges/zero only; no result | gamma/lambda_Q `not_found`; lambda_L `documented_but_unverified` | Blocked: gamma + lambda_Q + metric + timing |
| 16 | 548 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 24 | 540 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 32 | 532 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 40 | 524 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 48 | 516 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 56 | 508 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 64 | 500 | elementwise | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 8 | 556 | tensorial | projected | — | 0 (reported; not applied) | N/A | Selection count and gamma coefficient/ridge semantics unresolved | `35b5989` ranges/zero only; no result | gamma `not_found`; lambda_L `documented_but_unverified` | Blocked: gamma + metric + timing |
| 16 | 548 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 24 | 540 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 32 | 532 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 40 | 524 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 48 | 516 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 56 | 508 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 64 | 500 | tensorial | projected | — | 0 (reported; not applied) | N/A | Same | Same | Same | Blocked: gamma + metric + timing |
| 8 | 556 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Selection count and gamma/lambda_Q coefficient/ridge semantics unresolved | `35b5989` ranges/zero only; no result | gamma/lambda_Q `not_found`; lambda_L `documented_but_unverified` | Blocked: gamma + lambda_Q + metric + timing |
| 16 | 548 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 24 | 540 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 32 | 532 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 40 | 524 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 48 | 516 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 56 | 508 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |
| 64 | 500 | tensorial | inferred | — | 0 (coefficient and applied ridge) | — | Same | Same | Same | Blocked: gamma + lambda_Q + metric + timing |

The machine-readable counterpart is
`configs/1d/publication/figure4_parameter_evidence.json`. It is explicitly
non-executable and does not conform to the author-approved selected-parameter
contract.

## Figure 5 timing definition

The original executable script provides a generic per-model elapsed timer, but
no Figure 5 loop or speed-up calculation. Shared snapshot loading, the steady
solve, training selection, derivative construction, and POD are completed
before timers start. For each model, `time.time()` starts before case-specific
operator setup and stops only after initial-coordinate construction, ROM
integration, reconstruction, instantaneous error calculation, summary
printing, and (where applicable) nonlinear lifting or operator inference.
Plotting is absent. See `74bf785:542-649`; the corrected/pre-Phase-3 code has
the same boundary at `bff6910:732-839`.

This establishes only the historical integration-test timer:

- reconstruction, initial-coordinate optimization, operator construction,
  inference, error calculation, and printing are included;
- shared POD/training work is excluded;
- plotting is neither included nor timed;
- no FOM runtime denominator or speed-up is computed;
- no warm-up is discarded;
- no repeated-run average is computed (each model runs once).

It does **not** establish that the published Figure 5 online timings used this
boundary. Current Phase 5/5B timings are explicitly classified as
non-publication speed-up measurements and are not substituted. The Figure 5
timing definition therefore remains unresolved.

## Git-history provenance table

| Commit/artifact | File and relevant expression | What it establishes | Confidence |
|---|---|---|---|
| `74bf785` | `Nonlinear_Manifold_ROM.py:503-509`, `sqrt(square_err/square_inf)` | Instantaneous squared-error numerator, steady-state squared-norm denominator, then pointwise square root | `authoritative_executable_source` for the instantaneous metric only |
| `74bf785` | `Nonlinear_Manifold_ROM.py:535-540,654-668`, `errors.mean()` | Equal-weight full-history mean used only as the integration-test summary | `authoritative_executable_source`; not authoritative for Figures 4/5 |
| `74bf785` | `Nonlinear_Manifold_ROM.py:528-533` | One scalar `N_r=16`, `N_q=364` workflow and pre-scaled lifting/inference ridges; no rank loop or arrays | `authoritative_executable_source`; explicitly out of Figure 4 scope |
| `74bf785` | `Nonlinear_Manifold_ROM.py:161-169` | Original driver used 7,500 training snapshots | `authoritative_executable_source`; no Figure 4 selection tie |
| `bff6910` / `dd881b4` | `Nonlinear_Manifold_ROM.py:266-280` | Corrected/pre-Phase-3 inclusive training interval uses 7,501 snapshots; both commits share blob `997b531...` | `authoritative_executable_source`; no Figure 4 selection tie |
| `74bf785` / `bff6910` | Timer blocks described above | Exact generic integration-test timer boundary, with one run per case | `authoritative_executable_source`; Figure 5 applicability unverified |
| `35b5989` | `configs/1d/publication/experiments.json:542-567` | Figure 4 ranks, total dimension, candidate ranges, and reported `lambda_L=0`; no selections | `documented_but_unverified` |
| `35b5989` | `configs/1d/publication/README.md:46-53` | Explicitly states Figure 4 selections are unavailable; records Figure 5 stated constants | `documented_but_unverified` |
| `62712f0` | `one_d/rom.py:309-325,439-444` | Current publication parameters are coefficients scaled once by training count; legacy configuration values are already ridges | `authoritative_executable_source` for current semantics, not missing historical values |
| Phase 5B untracked result | `phase5b_report.md:145-148` | Original publication provenance unavailable and no Figure 4/5 study/search ran | `authoritative_saved_result` only for what Phase 5B did not run; no parameter evidence |

The original (`74bf785`), corrected Phase 1 (`bff6910`), and last pre-Phase-3
(`dd881b4`) scripts contain no latent-rank loop, lifting-rank loop,
rank-indexed regularization array, best-parameter dictionary, convergence
integral, speed-up formula, or commented Figure 4/5 plotting data. The
corrected and pre-Phase-3 commits use the identical script blob. No stale
comment, deleted file, or filename encodes a candidate selected value.

## Remaining author decisions

1. Provide the exact aggregate formula: numerator and denominator fields and
   powers, temporal rule and endpoint weights, final square root, and interval.
2. Provide the selected Figure 4 `gamma` for all 32 nonlinear cases and
   `lambda_Q` for the 16 inferred nonlinear cases, with selection provenance.
3. State whether each supplied value is a paper coefficient or an already
   scaled Gram ridge, and identify the training snapshot count used.
4. Define the published Figure 5 online timer boundary, the FOM denominator,
   and any warm-up/repetition policy.

The sigmoid initial condition is an author-confirmed figure-generation
configuration and is not an unresolved decision.
