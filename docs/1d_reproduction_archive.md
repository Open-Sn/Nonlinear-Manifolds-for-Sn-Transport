# One-dimensional reproduction archives

Phase 9 preserves the completed one-dimensional sigmoid benchmark in two
non-overlapping, deterministic archives. Archive creation validates and copies
existing artifacts only: it does not run a FOM, ROM, inference, derivative,
POD/SVD, search, metric, bundle construction, or plot.

## Archive roles

The core archive, rooted at `1d_reproduction/`, contains the exact committed
source as a Git bundle, source/environment metadata, the validated production
snapshot, all shared offline arrays, the independent tiny golden reference,
the five authoritative figure-data bundles, compact final-case JSON records,
the final Figure 1--5 renderings, configuration/provenance documents, and
complete checksummed inventories. It is the ordinary inspection and rerun
package.

The audit supplement, rooted at `1d_audit_supplement/`, contains Phase 5 and
5B reports, the Phase 7 execution plan, the complete Phase 8 search definition
and candidate history, rejected-candidate diagnostics, earlier diagnostic
plots, and superseded Figure 4/5 renders. It intentionally does not duplicate
the production snapshot, derivatives, POD arrays, authoritative bundles,
final cases, or final figures. The supplement depends on the core archive for
those scientific inputs and cannot rerun cases by itself.

Each archive has `README.md`, `inventory.json`, `inventory.tsv`,
`SHA256SUMS`, and `archive_metadata.json`. Inventories are sorted by relative
archive path and record the original repository-relative path, role, size,
checksum, generated/tracked status, source run ID, and authority. Any
repository-local absolute paths in text copies are rewritten only inside the
archive and recorded in `manifests/portable_path_map.json`; source artifacts
remain unchanged.

## Build and verify

Activate the project environment, validate both inclusion plans without
writing, and then create new outputs:

```bash
source ~/.bash_profile_june2024
python scripts/1d/build_reproduction_archives.py --kind both --dry-run
python scripts/1d/build_reproduction_archives.py --kind both
```

The builder pins every authoritative run in
`configs/1d/publication/archive_spec.json`, rejects a differing run ID unless
it is both explicitly supplied and explicitly allowed, checks core/audit
non-overlap, and refuses to overwrite either an archive or its checksum
sidecar. It uses deterministic `tar.gz` when `zstd` is unavailable. Generated
archives and `.sha256` sidecars are untracked under `dist/1d/`.

Verify each archive independently. A neighboring `.sha256` sidecar is used by
default:

```bash
python scripts/1d/verify_reproduction_archive.py \
  dist/1d/nonlinear-manifolds-1d-core-f2fb29b-20260802.tar.zst
python scripts/1d/verify_reproduction_archive.py \
  dist/1d/nonlinear-manifolds-1d-audit-f2fb29b-20260802.tar.zst
```

To retain a complete extraction, provide a new destination. The verifier
rejects existing destinations, absolute or parent-traversal names, links,
device entries, missing files, extra files, and checksum or inventory
disagreements:

```bash
python scripts/1d/verify_reproduction_archive.py <archive.tar.gz> \
  --extract-to <new-empty-destination>
```

## Restore source and scientific inputs

After verification and extraction, restore the committed source without
network access:

```bash
git clone 1d_reproduction/source/repository.bundle 1d-source
cd 1d-source
git checkout f2fb29b0fe7605dfbff0d42c7db552428c79876a
```

The scientific data can be used directly from the extracted core tree or
copied into a new working location. The essential portable locations are:

- `scientific_inputs/base_fom/` for the snapshot and FOM manifest;
- `scientific_inputs/shared_offline/` for time, indices, steady state,
  derivatives, POD basis, coefficients, and singular values;
- `scientific_inputs/shared_metric_inputs.npz` for the approved aggregate
  metric inputs;
- `configs/` for the catalog, selected Figure 4 parameters, initial-condition
  provenance, and metric/timing policy;
- `figure_data/figure1` through `figure_data/figure5` for authoritative compact
  bundles.

Use `inventory.json` rather than embedded source paths as the restoration map.
It associates every portable path with its original repository-relative
location. Verify again after any copy.

## Reproduce the presentation figures

Plot into new directories from the already validated bundles; no scientific
case execution or bundle construction is needed. For Figures 1--3, repeat the
command with the matching figure number:

```bash
python scripts/1d/plot_publication_figures.py \
  --figure 1 \
  --source-bundle <core>/1d_reproduction/figure_data/figure1 \
  --layout manuscript \
  --output-dir <new-figure1-directory>

python scripts/1d/plot_figure4.py \
  --source-bundle <core>/1d_reproduction/figure_data/figure4 \
  --output-dir <new-figure4-directory>

python scripts/1d/plot_figure5.py \
  --source-bundle <core>/1d_reproduction/figure_data/figure5 \
  --output-dir <new-figure5-directory>
```

The archived final outputs are already available in `final_figures/`; the
commands above are for presentation rerendering and refuse existing output
directories.

## Rerun final cases without a Figure 4 search

Restore `configs/` to their inventory-listed repository paths, then point the
case runner at the archived snapshot and shared-offline directory. The tracked
selected-parameter file resolves all 32 nonlinear Figure 4 cases directly;
ordinary final-case execution never consults the Phase 8 search directory.

```bash
python scripts/1d/run_publication_case.py \
  fig4_tensorial_inferred_nr32 \
  --snapshot <core>/1d_reproduction/scientific_inputs/base_fom/data/solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy \
  --shared-offline <core>/1d_reproduction/scientific_inputs/shared_offline \
  --output-root <new-results-root> \
  --execute
```

Use the analogous catalog case IDs preserved in `final_cases/` for Figures 2,
3, and 5. Always use a new result directory. Scientific runtimes and online
timings are machine-dependent; consult the final-case manifests rather than
treating recorded timing values as golden values.

## Repeat the Figure 4 search for audit only

Repeating the search is optional and is not part of ordinary reproduction.
First extract both archives, inspect
`phase8_search/search_definition.json`, and dry-run the resumable driver:

```bash
python scripts/1d/run_phase8_figure4.py \
  --run-id <new-run-id> \
  --snapshot <core-snapshot.npy> \
  --fom-manifest <core-fom-manifest.json> \
  --shared-offline <core-shared-offline-directory> \
  --shared-metric-inputs <core-shared-metric-inputs.npz> \
  --figure5-bundle <core>/1d_reproduction/figure_data/figure5 \
  --output-root <new-results-root> \
  --dry-run
```

Only add `--execute` after separately authorizing the scientific search and
reviewing its resource estimate. The audit supplement contains 1,384 completed
candidate/diagnostic records so most audits should inspect those records rather
than repeat the work.

## Storage, definitions, and limitations

The core is dominated by the approximately 480 MB production snapshot and the
approximately 421 MB shared-offline arrays; the audit supplement is much
smaller and does not duplicate them. Building and verifying are I/O-bound and
require temporary room for staging plus complete extraction. Exact planned and
compressed sizes are reported by the builder and stored in archive metadata.

The benchmark uses the author-confirmed localized sigmoid initial condition,
although the manuscript text states zero initial angular flux. Figures 4 and 5
use the author-approved `relative_space_time_l2_error_v1` metric and
`rom_solve_ivp_only_v1` solve-only online timing policy. The archived Figure 4
nonlinear parameters were regenerated by the documented Phase 8 sigmoid
search; they are not recovered historical manuscript parameters. Original
historical Figure 4/5 outputs, historical nonlinear selections, and their
source environment remain unavailable. These archives preserve the completed
repository regeneration and its audit trail; they do not claim exact
historical reproduction.
