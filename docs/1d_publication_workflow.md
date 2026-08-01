# Sigmoid-Based 1-D Publication Workflow

## Scope and benchmark policy

This layer describes publication-oriented experiments and future result
artifacts; it does not claim that the paper's numerical results have been
reproduced. Every case resolves directly to
`configs/1d/legacy_production.json` and its canonical snapshot
`solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy`.

The repository intentionally initializes the final positive angular block with

```text
1 - 1 / (1 + exp(-100 * (x - 0.1)))
```

using unit amplitude. The manuscript states zero initial angular flux. The
sigmoid is retained as a deliberate repository policy, likely to smooth the
beginning of the transient and reduce the sharp incompatibility between zero
flux and a suddenly imposed boundary inflow. This is a small intentional
deviation from the manuscript problem statement. It is not considered a
defect or a pending code correction, and no zero-initial-condition production
case is maintained.

Every catalog case, result manifest, figure-data bundle, and plot metadata must
record both `benchmark_variant=legacy_sigmoid` and the structured initial-
condition deviation.

## Experiment status

The catalog resolves to 57 cases in total.

| Figure | Cases | Benchmark | Specification status | Missing information | Execution readiness |
|---|---:|---|---|---|---|
| 1 | 1 POD study | Legacy sigmoid | Fully specified | Production snapshot | Ready once compatible snapshot exists |
| 2 | 3 projected ROMs | Legacy sigmoid | Fully specified | Production snapshot | Ready once compatible snapshot exists |
| 3 | 3 inferred ROMs | Legacy sigmoid | Fully specified | Production snapshot | Ready once compatible snapshot exists |
| 4 | 48 cases: 3 models × 2 operator types × 8 ranks | Legacy sigmoid | 16 linear partially specified; 32 nonlinear require author input | Linear: aggregate metric and timing; nonlinear: also selected gamma and inferred lambda_Q | Refused |
| 5 | 2 projected/inferred comparison studies | Legacy sigmoid | Requires author input | Aggregate metric and publication-comparable timing boundaries | Refused |

The catalog deterministically expands Figure 4 over linear, elementwise, and
tensorial models, projected and inferred operators, and all eight reported
`N_r` values. This produces all six series and 48 cases. Only nonlinear cases
have `N_q=564-N_r`; a linear model's dimension is simply `N_r`, and `N_q` is
not applicable. Linear projected cases carry no regularization. Linear
inferred cases carry only `lambda_L=0`; they do not use nonlinear lifting,
quadratic inference, nonlinear convergence tolerance, or alternating-
minimization iteration limits.

The linear dynamical models are fully parameterized but have status
`partially_specified` because the publication aggregate metric and timing
classification remain unresolved. Nonlinear Figure 4 cases additionally lack
selected per-case regularization. The reported regularization ranges are
search ranges, not selected values. The schema at
`configs/1d/publication/figure4_selected_parameters.schema.json` describes
future author-approved selections without supplying placeholders that could be
mistaken for results.

Figure 5 records `N_r=32`, all eight `N_q` values, and five distinct series:
fixed-rank linear, elementwise quadratic, tensorial quadratic, enlarged linear,
and the M-orthogonal best-projection benchmark. Parameters are attached only
to their scientific stages.

## Safe catalog commands

Listing and inspection are read-only and do not assemble production operators:

```bash
python scripts/1d/list_publication_cases.py
python scripts/1d/inspect_publication_case.py fig3_tensorial_inferred
python scripts/1d/inspect_publication_case.py fig4_tensorial_inferred_nr32
```

Running a case defaults to dry-run behavior:

```bash
python scripts/1d/run_publication_case.py fig3_tensorial_inferred --dry-run
```

An actual run requires all of the following:

1. A fully specified catalog case.
2. `--execute`.
3. The canonical, shape-compatible legacy sigmoid snapshot.
4. A new output directory; completed publication runs are immutable.

For example, after a compatible production dataset is available:

```bash
python scripts/1d/run_publication_case.py fig3_tensorial_inferred \
  --snapshot /path/to/solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy \
  --output-root results/1d/publication \
  --execute
```

Figure 4 and Figure 5 remain conservatively refused even with a snapshot
because their publication-comparable definitions are incomplete.

## Metrics

The implemented instantaneous error history is exactly

```text
e(t) = sqrt((d(t)^T M d(t)) / (psi_inf^T M psi_inf))
d(t) = psi_FOM(t) - psi_ROM(t)
```

It preserves the repository's mass convention and does not add angular
quadrature weights.

The Figure 4/5 time-aggregated relative convergence metric is intentionally not
implemented. Its numerator, denominator, squared-norm convention, temporal
quadrature, endpoint treatment, final square-root convention, and handling of
training versus extrapolation intervals require author input. The workflow
will not silently substitute a mean, RMS, trapezoidal or rectangle integral,
or mean instantaneous relative error.

Timing metadata separates online, offline, and total runtime and records the
speed-up basis plus included and excluded stages. The paper describes online
speed-up as excluding offline costs, but repository provenance does not yet
establish the exact boundaries used for every reported timing.

## Result artifacts and provenance

Actual authorized runs use:

```text
results/1d/publication/<case_id>/<run_id>/
├── config.json
├── case.json
├── manifest.json
├── metrics.json
├── diagnostics.json
├── data/
│   ├── pod_spectrum.npz
│   ├── fields.npz
│   ├── error_history.npz
│   └── convergence_data.npz
└── figures/
```

Only files relevant to a case are created. Full production trajectories are
not copied into publication artifacts. Figure 2/3 artifacts retain selected
fields at `t=2.5`, discrepancies, compact error histories, and diagnostics.

Manifests validate the case and catalog checksums, base configuration,
benchmark and deviation metadata, input snapshot checksum, model and operator
types, dimensions, regularization, solver state, training and output times,
metric definitions, timing classification, and every NPZ array's name, shape,
and dtype. JSON and named NPZ arrays are used instead of opaque pickles.

## Figure-data and plotting pipeline

Figure tools consume completed result directories only:

```bash
python scripts/1d/build_publication_figure_data.py \
  --figure 3 \
  --artifact results/1d/publication/<case>/<run> \
  --output-dir /path/to/figure3-data \
  --build

python scripts/1d/plot_publication_figures.py \
  /path/to/figure3-data \
  --output-dir /path/to/figure3-plots \
  --plot
```

Without `--build` or `--plot`, these commands report their planned actions and
write nothing. They never invoke FOM or ROM execution. Partial input sets are
marked `partial_input_set` and cannot be represented as complete publication
reproduction. Plot titles and metadata identify the legacy sigmoid benchmark
and carry the manuscript-deviation note.

## Current limitations

The production snapshot is not stored in this repository, the reorganized
workflow has not numerically reproduced any publication result, nonlinear
Figure 4 cases lack selected regularization provenance, and the Figure 4/5
aggregate metric and timing boundaries remain incomplete. A future production run must preserve the
Git, runtime, catalog, configuration, dataset checksum, benchmark-variant, and
manuscript-deviation provenance captured by the artifact schema.
