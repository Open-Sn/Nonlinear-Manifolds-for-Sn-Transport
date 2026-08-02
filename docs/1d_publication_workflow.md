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
authors confirm that the numerical calculations used to generate its
one-dimensional Figures 1--5 employed the localized sigmoid preserved here.
The repository therefore retains this configuration as the authoritative
numerical workflow for reproducing Figures 1--5. The manuscript-text
discrepancy remains explicit, but it is not a defect, workaround, pending code
correction, or unresolved question. No zero-initial-condition production case
is maintained.

Every future result manifest, figure-data bundle, and plot metadata must record
`benchmark_variant=legacy_sigmoid`, the structured manuscript-text discrepancy,
and `provenance_status=author_confirmed_figure_generation_configuration`.

## Experiment status

The catalog resolves to 57 cases in total.

| Figure | Cases | Benchmark | Specification status | Missing information | Execution readiness |
|---|---:|---|---|---|---|
| 1 | 1 POD study | Legacy sigmoid | Fully specified | Production snapshot | Ready once compatible snapshot exists |
| 2 | 3 projected ROMs | Legacy sigmoid | Fully specified | Production snapshot | Ready once compatible snapshot exists |
| 3 | 3 inferred ROMs | Legacy sigmoid | Fully specified | Production snapshot | Ready once compatible snapshot exists |
| 4 | 48 cases: 3 models × 2 operator types × 8 ranks | Legacy sigmoid | Fully specified for regenerated execution; historical selections unavailable | Production snapshot and execution assets | All 48 resolve through the catalog |
| 5 | 2 projected/inferred comparison studies | Legacy sigmoid | Fully specified | None | Ready |

The catalog deterministically expands Figure 4 over linear, elementwise, and
tensorial models, projected and inferred operators, and all eight reported
`N_r` values. This produces all six series and 48 cases. Only nonlinear cases
have `N_q=564-N_r`; a linear model's dimension is simply `N_r`, and `N_q` is
not applicable. Linear projected cases carry no regularization. Linear
inferred cases carry only `lambda_L=0`; they do not use nonlinear lifting,
quadratic inference, nonlinear convergence tolerance, or alternating-
minimization iteration limits.

The historical nonlinear Figure 4 selections remain unavailable, so reported
regularization ranges must not be mistaken for historical values. Phase 8
instead implements an author-approved regenerated search over the cataloged
ranges, including deterministic coarse grids, local geometric-midpoint
refinement, and an explicit tie rule. The reviewed values are now tracked in
`configs/1d/publication/figure4_selected_parameters.json`, labeled
`regenerated_sigmoid_search`, and linked by checksum from the catalog. The 16
linear cases remain directly specified; each of the 32 nonlinear cases resolves
its coefficients and applied ridges from that single tracked file. The catalog
keeps `complete_publication_reproduction=false` and explicitly records that the
values are regenerated rather than recovered historical parameters.

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

Individual nonlinear Figure 4 cases can now be inspected and dry-run directly;
these commands read only tracked files and do not assemble operators, solve a
ROM, write results, or consult the Phase 8 result directory:

```bash
python scripts/1d/inspect_publication_case.py fig4_tensorial_inferred_nr32
python scripts/1d/run_publication_case.py \
  fig4_tensorial_inferred_nr32 \
  --dry-run
```

With a compatible snapshot and shared-offline directory, the generic execution
path consumes the tracked selection without a search:

```bash
python scripts/1d/run_publication_case.py \
  fig4_tensorial_inferred_nr32 \
  --snapshot /path/to/solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy \
  --shared-offline /path/to/validated/shared-offline \
  --output-root results/1d/publication \
  --execute
```

The portable catalog/configuration is sufficient in a fresh clone; the
original `results/1d/publication/phase8_runs/` tree is not required. Figure 4
and Figure 5 still have separate resumable study plans. Repeating the Figure 4
search remains possible for verification by reviewing the Phase 8 dry-run:

```bash
python scripts/1d/run_phase8_figure4.py \
  --run-id <new-run-id> \
  --snapshot <production-snapshot.npy> \
  --fom-manifest <validated-fom-manifest.json> \
  --shared-offline <shared-offline-directory> \
  --shared-metric-inputs <phase7-shared-metric-inputs.npz> \
  --figure5-bundle <validated-phase7-figure5-bundle> \
  --dry-run
```

Add `--execute` only when intentionally verifying the search and after recording
resource availability and the runtime estimate. Phase 8 writes its checksummed
search definition before initializing scientific execution, reuses completed
candidates on resume, and promotes selected nonlinear candidates into
final-case manifests without a duplicate solve. Ordinary regenerated Figure 4
execution does not invoke this driver. The driver does not run the FOM,
derivatives, POD/SVD, or Figure 5.

The Phase 7 Figure 5 plan is:

```bash
python scripts/1d/run_phase7_figure5.py \
  --run-id <new-run-id> \
  --snapshot <production-snapshot.npy> \
  --fom-manifest <validated-fom-manifest.json> \
  --shared-offline <shared-offline-directory> \
  --dry-run
```

Add `--execute` only after reviewing that plan. Successful per-case directories
are reused on resume; failed cases retain diagnostics, and interrupted cases
remain pending. The command never runs the FOM, recomputes derivatives, or
recomputes the POD/SVD.

## Metrics

The implemented instantaneous error history is exactly

```text
e(t) = sqrt((d(t)^T M d(t)) / (psi_inf^T M psi_inf))
d(t) = psi_FOM(t) - psi_ROM(t)
```

It preserves the repository's mass convention and does not add angular
quadrature weights.

Figures 4 and 5 use the author-approved repository metric

```text
sqrt(trapezoid((FOM-ROM)^T M (FOM-ROM), time)
     / trapezoid(FOM^T M FOM, time))
```

over the complete `[0,10]` interval, including both endpoints. Its denominator
is the uncentered transient FOM and the established mass convention is retained
without angular weights. This definition was adopted for regeneration; it was
not recovered from historical executable source.

Figure 4 and Figure 5 online times are wall-clock times inside the reduced
`solve_ivp` call only. Speed-up divides the exact validated FOM-manifest
integration time by that online value. Every excluded setup, inference,
reconstruction, metric, and artifact stage is timed separately. One run, no
warm-up, and no repeated average are required; the resulting speed-ups are
machine-specific.

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

The retained positional form produces the earlier diagnostic layout. The
bundle-only manuscript layout for Figures 1--3 uses an explicit figure and
source bundle:

```bash
python scripts/1d/plot_publication_figures.py \
  --figure 3 \
  --source-bundle /path/to/validated-figure3-data \
  --layout manuscript \
  --output-dir /path/to/new/manuscript-figure3 \
  --dry-run

python scripts/1d/plot_publication_figures.py \
  --figure 3 \
  --source-bundle /path/to/validated-figure3-data \
  --layout manuscript \
  --output-dir /path/to/new/manuscript-figure3
```

Diagnostic mode writes only with `--plot`; manuscript `--dry-run` validates
and reports without writing. Neither mode invokes FOM or ROM execution.
Manuscript mode accepts only a complete matching Figure 1--3 bundle and
refuses an existing output directory. Partial input sets are marked
`partial_input_set` and cannot be represented as complete publication
reproduction. Plot metadata identifies the legacy sigmoid benchmark, carries
the manuscript-text discrepancy and author-confirmed figure-generation
provenance, and records the physical phase-space mapping.

Figure 4 has dedicated bundle-only commands:

```bash
python scripts/1d/build_figure4_bundle.py \
  --run-id <phase8-run-id> \
  --output-dir <new-bundle-directory> \
  --build

python scripts/1d/plot_figure4.py \
  --source-bundle <new-bundle-directory> \
  --output-dir <new-plot-directory> \
  --dry-run

python scripts/1d/plot_figure4.py \
  --source-bundle <new-bundle-directory> \
  --output-dir <new-plot-directory>
```

The plot validates and consumes only the compact 48-case bundle. It writes a
2-by-2 PNG/PDF composite, metadata JSON, and caption Markdown and cannot launch
a solver.

Figure 5 has corresponding bundle-only commands:

```bash
python scripts/1d/build_figure5_bundle.py \
  --run-id <run-id> \
  --output-dir <new-bundle-directory> \
  --build

python scripts/1d/plot_figure5.py \
  --source-bundle <new-bundle-directory> \
  --output-dir <new-plot-directory> \
  --dry-run

python scripts/1d/plot_figure5.py \
  --source-bundle <new-bundle-directory> \
  --output-dir <new-plot-directory>
```

The plot command validates and reads the compact bundle only; it cannot launch
a scientific solve. It writes a 2-by-2 PNG/PDF composite, metadata JSON, and a
caption Markdown file, and refuses to overwrite an existing directory.

## Current limitations

The original figure-generating source commit, dependency environment, and
historical nonlinear Figure 4 selections are not known. Phase 8 values are
regenerated under an author-approved current protocol and are tracked so that
the regenerated study can be rerun without searching again. The tracked file is
not evidence of the original selections. The approved aggregate metric and
timing boundaries are explicit repository regeneration definitions, not
recovered historical provenance. A future production run must preserve the
Git, runtime, catalog, configuration, selected-parameter, dataset checksum,
benchmark variant, and manuscript-deviation provenance captured by the artifact
schema.
