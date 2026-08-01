# Open scientific and provenance questions

The Phase 1 verification work protects current behavior while leaving the
following scientific and publication-provenance questions unresolved.

1. The paper states zero initial angular flux, while the current production
   driver uses a localized sigmoid initial condition in one angular block.
2. The production sigmoid is intentionally preserved during the current
   software-verification phase pending provenance and author review.
3. The paper uses $N_r=16$, $N_q=548$ for an illustrative case and
   $N_r+N_q=564$ for a convergence study. The current executable defaults
   differ and require author confirmation; Phase 1 does not change them.
4. The exact production snapshot used for the paper has not yet been
   identified in this repository.
5. The source commit and dependency environment associated with the
   published figures have not yet been established.
6. The current error routine computes an instantaneous mass-norm error
   normalized by the steady-state norm. The paper's convergence figures also
   require a relative time-integrated error, for which a publication
   reproduction pipeline is not yet implemented.
7. The current algebraic mass norm does not add angular quadrature weights.
   This convention is preserved in Phase 1 and should be reviewed separately.
8. Exact regularization values and their scaling for every published case
   should be tied to explicit run configurations in a later phase.
9. Publication timing and speed-up measurements require a documented
   definition of which offline costs are excluded.

These items are not resolved by the software-verification tests. They remain
preserved pending provenance review and require author confirmation.
