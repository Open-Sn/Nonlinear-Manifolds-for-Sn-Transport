# 1-D Figure 4/5 provenance and approved regeneration policy

The Phase 6 history audit found no historical executable implementation of
the aggregate Figure 4/5 error, no historical Figure 5 speed-up loop, and no
selected nonlinear Figure 4 regularization values. Phases 7 and 8 do not revise
those findings. Instead, an author of the manuscript approved explicit
repository definitions for regenerating the sigmoid-benchmark Figure 4/5
metric and online timing. These definitions are not described as historically
recovered.

The authors also confirm that the one-dimensional figure calculations used the
localized sigmoid workflow in `configs/1d/legacy_production.json`, despite the
manuscript text stating zero initial angular flux. The benchmark identity
remains `legacy_sigmoid` with provenance status
`author_confirmed_figure_generation_configuration`.

## Approved aggregate error

For Figures 4 and 5, the repository definition
`relative_space_time_l2_error_v1` is

\[
E_{rel}=\sqrt{
\frac{\int_0^{10}(\psi_{FOM}-\psi_{ROM})^T M
(\psi_{FOM}-\psi_{ROM})\,dt}
{\int_0^{10}\psi_{FOM}^T M\psi_{FOM}\,dt}}
.
\]

Both integrals use the composite trapezoidal rule and include `t=0` and
`t=10`. The denominator is the full uncentered transient FOM, not the steady
state or centered transient. The final square root is required. `M` retains
the global algebraic mass convention already used by the code, with no added
angular quadrature weights.

This metric is used consistently for integrated ROMs, enlarged-linear ROMs,
and the affine best-M-projection benchmark. It is not an arithmetic mean of
instantaneous errors or an RMS of the Figure 2/3 error. The existing
instantaneous steady-state-normalized metric for Figures 2 and 3 is unchanged.

Historical executable source only establishes the older instantaneous metric

\[
e_\infty(t_i)=\sqrt{d(t_i)^T M d(t_i)/(\psi_\infty^T M\psi_\infty)}.
\]

It therefore remains evidence about historical behavior, not provenance for
the approved aggregate definition.

## Approved online timing and speed-up

The repository definition `rom_solve_ivp_only_v1` measures wall-clock time
only inside each reduced `solve_ivp` call. The following stages are recorded
separately and excluded from the online denominator: lifting construction,
projected-operator construction, inference, nonlinear initial-coordinate
fitting, reconstruction, metric evaluation, artifact writing, and all shared
offline work.

Speed-up is

```text
validated production FOM integration elapsed time
-------------------------------------------------
reduced solve_ivp wall-clock elapsed time
```

The numerator is read at full precision from the validated FOM manifest. One
measured run per case is sufficient; no warm-up or repeated-run average is
required. Timings are machine-specific observations, not scientific golden
values. Each result records platform, CPU, Python, NumPy, SciPy, BLAS, `nfev`,
`njev`, and `nlu` metadata.

The original integration-test script used a broader generic timer that also
included model setup, reconstruction, and error work. Phase 7 does not rename
or reinterpret that historical timer.

## Figure 4 readiness

The approved metric and timing decisions complete the definitions needed by
the 16 linear Figure 4 cases. Those cases are cataloged as fully specified:
linear projected cases have no regularization, while linear inferred cases use
`lambda_L=0` and a closed-form inference solve.

No authoritative historical rank-specific `gamma` or inferred `lambda_Q`
values were found. The machine-readable historical audit at
`configs/1d/publication/figure4_parameter_evidence.json` therefore remains
unchanged and continues to document the failed recovery. It is not replaced or
reinterpreted by regenerated values.

Phase 8 supplies a separate author-approved regenerated protocol. For every
nonlinear model and rank it evaluates nine geometrically spaced `gamma`
coefficients from `7e-10` through `5e-5` with projected operators, followed by
geometric-midpoint refinement to the available coarse neighbors. The selected
projected `gamma` is reused for the corresponding inferred lifting. Inference
then evaluates nine geometrically spaced `lambda_Q` coefficients from `6e-9`
through `2e-4` and applies the same local refinement. Coefficients are
multiplied by the validated `N_s=7501` training count exactly once. The
objective is `relative_space_time_l2_error_v1`; candidates within 0.1 percent
of the minimum are tied and the larger regularization coefficient wins.
`lambda_L=0`, the `1e-6` inference tolerance, and the 100000-iteration limit
are fixed.

The definition is checksummed before execution, candidate results are compact
and resumable, and final nonlinear artifacts reference selected search
candidates rather than rerunning them. These outputs are labeled
`regenerated_sigmoid_benchmark` with selection provenance
`regenerated_sigmoid_search`. They are not recovered historical parameters or
an exact reproduction of the original Figure 4 calculation.

After review, the selected coefficients and complete selection metadata were
promoted to the portable tracked file
`configs/1d/publication/figure4_selected_parameters.json`. Its deterministic
checksum, strict schema, per-case applied ridges, tie outcomes, and source
checksums allow the catalog to resolve all 32 nonlinear cases without reading
`results/1d/publication/phase8_runs/`. Together with the 16 directly specified
linear cases, all 48 regenerated-sigmoid Figure 4 cases are now execution-ready
once a valid production snapshot and required shared-offline inputs are
available. Catalog resolution and individual execution consume the tracked
values directly and never invoke the search.

This portability changes regenerated-study readiness, not the historical
provenance conclusion. A fresh clone can rerun the regenerated Figure 4 study
without repeating the regularization search; the Phase 8 search can still be
repeated independently to verify the documented selection protocol.

## Figure 5 readiness

Both aggregate Figure 5 studies are fully specified and executable. They use
`N_r=32`, `N_q=[1,2,4,8,16,32,64,128]`, the fixed and enlarged linear
comparators, elementwise and tensorial nonlinear models, and the shared affine
best-M-projection benchmark. No regularization search is part of Figure 5.

Publication coefficients are scaled by the validated `N_s=7501` training
count exactly once:

| Model/stage | Coefficient | Applied Gram ridge |
|---|---:|---:|
| Elementwise lifting `gamma` | `8e-7` | `0.0060008` |
| Elementwise inferred `lambda_Q` | `8e-7` | `0.0060008` |
| Tensorial lifting `gamma` | `2.5e-8` | `0.000187525` |
| Tensorial inferred `lambda_Q` | `4e-7` | `0.0030004` |
| Inferred linear `lambda_L` | `0` | `0` |

## Remaining provenance limits

- Original historical Figure 4/5 result files and their source environment are
  unavailable.
- Historical nonlinear Figure 4 selected regularization and selection
  provenance remain unavailable; Phase 8 selections are regenerated under the
  explicit protocol above.
- Regenerated Figure 5 results use the corrected current workflow and the
  author-approved metric/timing policy, so they are not claimed as exact
  historical reproduction.
- The sigmoid/manuscript-text discrepancy remains explicit in every artifact.
