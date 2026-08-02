# Comparison of regenerated one-dimensional Figures 1--3

## Scope and evidence

The manuscript-style figures are new renderings of the completed, validated
`legacy_sigmoid` bundles. The comparison below addresses scientific content
that the arrays support. It does not claim bit-for-bit image identity or exact
historical software provenance. The author confirmation establishes that the
manuscript's one-dimensional Figures 1--3 used the localized sigmoid workflow;
the original figure-generating commit, environment, and separately archived
publication dataset remain unavailable.

## Figure 1: POD-energy decay

The unresolved energy falls from
$E_{\mathrm{unres}}(16)=6.7992242523\times10^{-3}$ to the numerical floor
$E_{\mathrm{unres}}(564)=5.5511151231\times10^{-16}$. This supports the
manuscript construction with $N_r=16$ latent coordinates and $N_q=548$
lifting coordinates. The scientific values and marked dimensions are
**numerically consistent** with the validated bundle. Restricting the display
to dimensions 0--700 and using shaded latent/lifting regions is **visually
reformatted but scientifically equivalent**. Pixel-level agreement with the
historical Figure 1 is **not directly verifiable**.

## Figures 2 and 3: representative errors

The table reports the stored instantaneous steady-state-normalized mass error.

| Operators | Model | $t=2.5$ | $t=7.5$ | $t=10$ |
|---|---|---:|---:|---:|
| Projected | Linear | 5.894298e-2 | 2.287111e-2 | 9.055414e-3 |
| Projected | Polynomial (elementwise) | 5.677924e-2 | 2.425492e-2 | 9.817161e-3 |
| Projected | Tensorial | 3.446520e-2 | 1.610939e-2 | 5.725624e-3 |
| Inferred | Linear | 5.894298e-2 | 2.287111e-2 | 9.055413e-3 |
| Inferred | Polynomial (elementwise) | 5.819484e-2 | 2.203431e-2 | 8.785251e-3 |
| Inferred | Tensorial | 3.433110e-2 | 3.147877e-3 | 4.211214e-4 |

The projected Linear and Polynomial histories are close: Polynomial is lower
at $t=2.5$ but slightly higher at the training boundary and final time. Their
field discrepancies at $t=2.5$ also have comparable maximum magnitudes
(0.1493 and 0.1461). This is **qualitatively consistent** with the reported
linear-versus-elementwise similarity.

The projected Tensorial model has the lowest error at all three representative
times. Its maximum field-discrepancy magnitude at $t=2.5$ is 0.1326, compared
with 0.1493 for Linear. This accuracy improvement is **numerically consistent**
with the stored Phase 5B results.

Projected and inferred Linear results agree to approximately ten decimal
places at the representative times, as expected for the corresponding linear
construction. The inferred Polynomial result remains in the same error range
as Linear, with a modest advantage at $t=7.5$ and $t=10$. These comparisons
are **numerically consistent** with the bundles and **qualitatively
consistent** with similar Linear and elementwise behavior.

The inferred Tensorial history separates most strongly after the early-time
region: its error decreases from 3.433110e-2 at $t=2.5$ to 3.147877e-3 at the
training boundary and 4.211214e-4 at $t=10$. At final time this is about 21.5
times smaller than inferred Linear and 20.9 times smaller than inferred
Polynomial. Its strictly decreasing late-time scale and bounded completion to
$t=10$ support **numerically consistent** extrapolation stability and an
inferred-Tensorial improvement. They do not prove stability outside the stored
interval.

## Layout and provenance differences

The regenerated Figures 2 and 3 replace flattened phase-space indices with
four labeled ordinate curves over physical $x\in[0,3]$, preserve duplicate DG
interface endpoints, use one scale shared by all field panels, and use a
separate symmetric scale for each discrepancy panel. The error histories are
combined into a full-width logarithmic panel with black $t=2.5$ and red
$t=7.5$ markers. These choices are **visually reformatted but scientifically
equivalent** to the validated arrays. Exact fonts, spacing, colors, crop, and
historical pixels are **not directly verifiable**.

The manuscript's zero-initial-flux statement differs from the confirmed
figure-generating calculation. The sigmoid configuration itself is
author-confirmed; the degree of visual identity between these renderings and
the historical published images remains **not directly verifiable**.
