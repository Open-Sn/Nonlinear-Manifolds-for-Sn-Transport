# Open scientific and provenance questions

The manuscript states zero initial angular flux, but the authors confirm that
the numerical calculations used for its one-dimensional figures employed
the localized sigmoid in the final positive angular block. The authoritative
configuration is `configs/1d/legacy_production.json`; the repository retains
it as the reproduction workflow for Figures 1--5. This transparent text/code
discrepancy is resolved provenance, not an open question or pending code fix.
The remaining scientific and publication-provenance questions are:

1. The paper uses $N_r=16$, $N_q=548$ for an illustrative case and
   $N_r+N_q=564$ for a convergence study. The current executable defaults
   differ; publication cases now record the paper-oriented dimensions without
   changing the historical defaults.
2. The source commit and dependency environment associated with the
   published figures have not yet been established.
3. The current algebraic mass norm does not add angular quadrature weights.
   This convention is preserved in Phase 1 and should be reviewed separately.
4. Historical Figure 4 selected nonlinear regularization values and their
   selection provenance are unavailable. Phase 8 addresses current
   regeneration with a separate author-approved, checksummed search; it does
   not recover this historical provenance.

The aggregate Figure 4/5 metric and Figure 5 online timing boundary are no
longer open questions. An author approved explicit repository definitions for
sigmoid-benchmark regeneration. Those decisions are documented in
`docs/1d_figure4_5_provenance.md` and are not presented as historical-source
recovery.

These historical items are not resolved by software-verification tests. They
remain preserved as limitations even when the regenerated studies complete.
