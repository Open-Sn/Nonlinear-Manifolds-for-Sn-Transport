# Repository captions for one-dimensional Figures 1--3

These captions describe the manuscript-style plots regenerated from the
validated `legacy_sigmoid` figure-data bundles. They are repository captions,
not transcriptions of the manuscript.

## Figure 1: POD reducibility

Relative unresolved POD energy as a function of retained/reconstruction
dimension for the sigmoid benchmark. The latent state retains
$N_r=16$ modes; the nonlinear reconstruction adds $N_q=548$ lifting modes,
giving total dimension $N_r+N_q=564$. The shaded intervals distinguish the
latent and lifting portions. The unresolved-energy values at dimensions 16
and 564 are annotated on the curve.

## Figure 2: projected reduced models

Full-order reference and projected Linear, Polynomial (elementwise
quadratic), and Tensorial reduced-model angular fluxes for the sigmoid
benchmark. The top and middle rows show the four ordinate fields and signed
reference-minus-reconstruction discrepancies at $t=2.5$. All reduced models
use $N_r=16$; the nonlinear models additionally use $N_q=548$. The bottom
panel gives the complete instantaneous steady-state-normalized error history
on $0\leq t\leq10$, with the displayed-field time at $t=2.5$ and the
training/extrapolation boundary at $t=7.5$.

## Figure 3: inferred reduced models

Full-order reference and inferred Linear, Polynomial (elementwise quadratic),
and Tensorial reduced-model angular fluxes for the sigmoid benchmark. The
four-ordinate fields and signed reference-minus-reconstruction discrepancies
are shown at $t=2.5$. All models use $N_r=16$, and the nonlinear models use
$N_q=548$. The combined error panel spans $0\leq t\leq10$ and marks
$t=2.5$ and the $t=7.5$ training/extrapolation boundary. Inference convergence
and applied-ridge details are retained in the associated plot metadata rather
than placed on the scientific panels.

## Provenance note

The manuscript text states zero initial angular flux; the authors confirm
that the displayed one-dimensional manuscript results were generated using
the localized sigmoid initialization represented here and preserved in
`configs/1d/legacy_production.json`.
