# One-dimensional configurations

`legacy_production.json` records every important default used by the current
root-level 1-D FOM and ROM scripts. It is a behavior-preservation configuration,
not a claim that the repository currently reproduces the publication figures.

In particular, it explicitly preserves the localized sigmoid initial condition
in the final angular block, with unit amplitude, transition at `x=0.1`, and
steepness `100`. The paper states a zero initial condition. Resolving that
scientific discrepancy is outside Phase 3, so no alternative paper-zero
production configuration is supplied here.

The physical inflow selection is named `most_normal`: for the ascending GL4
ordinates this is the final positive ordinate. The historical root helper calls
that same final ordinate `"most_grazing"`; the compatibility behavior is not
changed.

All historical ROM constants are explicit, including the otherwise inactive
linear-case quadratic regularization (`1e-5`), the elementwise value (`12`),
the tensorial value (`0.012`), and the lifting regularization (`0.0001875`).

Configurations are loaded and validated by `one_d.config.OneDConfig`. The
canonical JSON serialization and SHA-256 configuration checksum are
deterministic and are recorded in provenance manifests.
