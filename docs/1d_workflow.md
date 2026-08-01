# Organized one-dimensional workflow

Phase 3 adds explicit configuration, reusable execution stages, safe command
lines, and provenance-aware output directories around the existing 1-D code. It
does not change the scientific model and does not claim publication
reproduction.

## Repository organization

```text
one_d/
  config.py       typed configuration, validation, canonical JSON, checksum
  problem.py      configured mesh/problem construction and DG assembly
  fom.py          time grid, FOM solve/validation, snapshot save/load/inspection
  rom.py          selected-case ROM preparation, operators, solve, reconstruction
  workflows.py    dry-run and explicitly authorized execution orchestration
  provenance.py   run-directory and manifest creation/update

configs/1d/
  legacy_production.json
  README.md

scripts/1d/
  generate_fom.py
  run_rom.py
  inspect_snapshot.py

results/1d/
  README.md

Transport_Driver_Benchmark_1D.py   historical FOM entry point
Nonlinear_Manifold_ROM.py          historical six-case ROM entry point
```

The root scripts remain supported with their historical defaults:

```bash
python Transport_Driver_Benchmark_1D.py
python Nonlinear_Manifold_ROM.py
```

They retain their established output and terminal summaries. The organized
scripts are safer for deliberate new work because execution requires
`--execute`; their dry-run and inspection output is structured JSON.

## Four distinct scopes

1. **Legacy production configuration** records current executable defaults,
   including the localized sigmoid. It is the configuration for preserving
   repository behavior.
2. **Tiny verification tests** exercise small production-implementation paths
   quickly; they are not scientific results.
3. **Independent golden references** in `tests/golden/` use a separate dense DG,
   matrix-exponential, and correlation-POD implementation to protect tiny-case
   numerical behavior.
4. **Publication reproduction** is future work. No current configuration or
   result directory is claimed to reproduce published figures or tables.

## Current interfaces and defaults

The committed root FOM initializes the GL4 quadrature, 750-cell mesh, material
operators, mass matrices, streaming operator, boundary map, steady source, and
mass transforms through `initialize_production_problem()`. Its `main()` checks
the canonical current-working-directory snapshot path, reuses an existing file,
or calls the Radau FOM and saves `sol.y`.

The root ROM class exposes snapshot validation and steady-state computation in
`load_training_data()`, physical training-window selection, eighth-order
derivatives, mass-weighted POD, nonlinear lifting, projected operators, inferred
operators, initial conditions, integration, reconstruction, and the existing
normalized mass-matrix error. The root `main()` still runs the historical six
cases and prints per-case and summary errors.

The complete executable defaults are in
`configs/1d/legacy_production.json`, including:

- widths `[1,1,1]`, 250 cells per region, GL4;
- `sigma_t=[0,1,0]`, `sigma_s=[0,0.99,0]`;
- unit left inflow on the physical most-normal positive ordinate;
- the protected final-block sigmoid with transition `0.1` and steepness `100`;
- FOM Radau tolerances `atol=1e-9`, `rtol=1e-12`;
- ROM Radau tolerances `atol=1e-12`, `rtol=1e-9`;
- `N_r=16`, `N_q=364`, and all current regularizations;
- nonlinear-inference tolerance `1e-6` and finite limit `100000`;
- the 10,001-point grid through time 10 and training through time 7.5;
- canonical snapshot
  `solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy`.

## Inspect the production configuration

Configuration loading validates geometry, materials, time metadata, dimensions,
embedding/operator selections, finite values, and consistency between the
filename and implied angle/cell/time counts.

The configuration can be inspected as ordinary JSON:

```bash
python -m json.tool configs/1d/legacy_production.json
```

Its canonical serialization and SHA-256 checksum are recorded in every new run
manifest.

## Safe FOM dry run

```bash
python scripts/1d/generate_fom.py \
  --config configs/1d/legacy_production.json \
  --dry-run
```

The dry run reports expected dimensions, shape, path, approximately allocated
raw snapshot bytes, existence, compatibility, and whether execution would solve,
reuse, overwrite, or refuse. It does not construct matrices, integrate, or
write. Omitting both `--dry-run` and `--execute` also defaults safely to this
read-only report.

## Deliberate FOM execution

Only an explicit flag enables scientific work:

```bash
python scripts/1d/generate_fom.py \
  --config configs/1d/legacy_production.json \
  --output-dir results/1d/my-run \
  --execute
```

This production command is expensive and was not run during Phase 3. Existing
snapshots are shape/finiteness checked before reuse. Overwriting requires
`--overwrite` or explicit permission in configuration.

## Selected ROM dry run and execution

Inspect one choice without loading data or assembling operators:

```bash
python scripts/1d/run_rom.py \
  --config configs/1d/legacy_production.json \
  --model tensorial \
  --operators inferred \
  --dry-run
```

Models are `linear`, `elementwise` (also accepted as `element-wise`), and
`tensorial`. Streaming operators are `projected` or `inferred`.

Execute exactly one selected case deliberately with:

```bash
python scripts/1d/run_rom.py \
  --config configs/1d/legacy_production.json \
  --model tensorial \
  --operators inferred \
  --input-snapshot solutionDG1_A4_T10_Nt10001_Nx750_continuous_bis.npy \
  --output-dir results/1d/tensorial-inferred \
  --execute
```

The historical six-case sequence remains available through the root ROM script.
It is never started merely by importing a module or running a new dry run.

## Snapshot inspection

```bash
python scripts/1d/inspect_snapshot.py path/to/snapshot.npy \
  --config configs/1d/legacy_production.json \
  --sha256
```

Inspection reports path, file bytes, array shape and dtype, finiteness, expected
shape, compatibility, optional content SHA-256, and configured time/raw-memory
metadata. It never modifies the file.

## Run directories and provenance

Explicit organized executions create:

```text
results/1d/<run_id>/
  config.json
  manifest.json
  logs/
  data/
  metrics/
  figures/
```

The manifest records run identity and UTC dates, Git commit and dirty status,
Python/NumPy/SciPy/platform versions, configuration source and canonical content,
configuration checksum, snapshot metadata and optional content hash, execution
stage and solver success, timing, diagnostics, and parent/input provenance. A
dirty tree is allowed but recorded. Tests create run contracts only under pytest
temporary directories.

## Why the sigmoid remains

The current driver initializes only the final angular block with
`1 - 1/(1 + exp(-100*(x-0.1)))`. Phase 3 makes that behavior explicit rather
than silently replacing it. The paper states a zero initial condition; resolving
the discrepancy requires a separate scientific decision and validation effort.
Consequently, the legacy configuration is not yet asserted to be the exact
publication configuration.
