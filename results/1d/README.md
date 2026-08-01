# One-dimensional run outputs

Explicit Phase 3 executions create provenance-aware run directories here by
default:

```text
results/1d/<run_id>/
  config.json
  manifest.json
  logs/
  data/
  metrics/
  figures/
```

Dry runs never create these directories. Production data, metrics, and figures
are intentionally not committed. Only this contract documentation and
`.gitkeep` are tracked.
