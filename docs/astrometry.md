# Astrometric microlensing

`singlens` computes the magnification and astrometric centroid shift of the
same single-lens event, including finite sources and linear limb darkening.
The astrometric result describes the unresolved source images. To predict an
observed centroid, scale it by the angular Einstein radius, choose sky
coordinates, and combine it with the flux and position of any blended light.

## What the shift represents

Let $\boldsymbol\theta_s(t)$ be the unlensed source center and
$\boldsymbol\theta_{\mathrm{img}}(t)$ the flux-weighted centroid of its lensed
images. The source-only astrometric shift is

$$
\boldsymbol\delta_s(t)
=\boldsymbol\theta_{\mathrm{img}}(t)-\boldsymbol\theta_s(t).
$$

`sl.astrometric_shift` returns $\boldsymbol\delta_s/\theta_E$ as a complex
array. Its real and imaginary components are along the trajectory's
$\tau$ and $\beta$ axes, respectively. For a static event,

$$
w(t)=\tau(t)+i\beta(t)=\frac{t-t_0}{t_E}+iu_0.
$$

These are source positions relative to the lens in Einstein-radius units.
The output is a displacement relative to the source, not the image centroid
relative to the lens and not an absolute sky position. For a point source,

$$
\frac{\delta_s(t)}{\theta_E}=\frac{w(t)}{|w(t)|^2+2}.
$$

Multiply by $\theta_E$ in mas to obtain a complex displacement in mas;
use `.real` and `.imag` to access its two signed components. Multiplying
the complex result by $e^{i\phi}$ rotates both components into a chosen
sky frame. The default axes are not RA and Dec.

For an extended source, the centroid depends on both the magnification and
the first position moment of the images. `singlens` convolves these
quantities separately with the source brightness and then divides the
moment by the flux. Circular symmetry makes the two convolutions radial:
their Hankel transforms use $J_0$ and $J_1$, respectively. The
[microlensing model](theory/index.md#astrometric-centroid-shift) gives the
equations and their FFTLog implementation.

The finite-source construction follows
[Mao & Witt (1998)](https://arxiv.org/abs/astro-ph/9804045) and
[Lee et al. (2010)](https://arxiv.org/abs/1005.3021). In particular, a source
transit can reverse the shift's direction and deform the point-source
ellipse. Averaging point-source shifts alone would miss the redistribution
of light across the source.

## Light curve and centroid of the same event

The following illustrative event has $t_E=20$ days, $u_0=0.1$, and
$\theta_E=1.2$ mas. The finite sources have radius $\rho=0.5$, so the lens
crosses their disks and the source-size effects are easy to see.
Three curves use the same trajectory: a point source, a uniform disk, and
a linearly limb-darkened disk with `a1=0.6`.

The time-series presentation follows the shared time axes and limb-contact
markers in [Lee et al. (2010), Figs. 5–6 and 8](https://arxiv.org/pdf/1005.3021).
The curves below are new `singlens` point-lens calculations. The paper also
studies finite-lens occultation, which is outside this example's model.

```python
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import singlens as sl
```

Prepare one `Model` for each source profile and evaluate its photometry
and astrometry with the same event parameters:

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN SOURCE EXAMPLE
:end-before: END SOURCE EXAMPLE
:dedent: 4
```

Here `a1` is the coefficient in $I(\mu)=1-a_1(1-\mu)$, where
$\mu=\sqrt{1-(r/\rho)^2}$. It is not the flux-normalized coefficient
$\Gamma=2a_1/(3-a_1)$. In the event-level functions `sl.magnification`
and `sl.astrometric_shift`, omit `rho` for a point source and omit `a1`
for a uniform disk.

```{figure} ../example/astrometry/outputs/source_profiles.png
:alt: Magnification and two signed centroid-shift components in three vertically aligned panels sharing one time axis.

Magnification and source-only centroid shifts on a common time axis, from
$t_0-40$ to $t_0+40$ days. The vertical dotted lines in every panel mark
$u=\rho$, at $t-t_0=\pm t_E\sqrt{\rho^2-u_0^2}\simeq\pm9.80$ days.
Shifts are in $\mu$as in the trajectory frame; the example code computes
them in mas and converts them for plotting. Line styles distinguish the
profiles as well as colors.
```

Finite source size changes both the peak light curve and the signed
astrometric components. A uniform disk has zero shift when $u=\rho$ as
well as at exact alignment; the astrometric behavior near contact depends
on the brightness profile.

```{figure} ../example/astrometry/outputs/source_tracks.png
:alt: Separate two-dimensional centroid-shift tracks comparing the point-source ellipse with the finite-source loops.

The same event in the $(\delta_{s,\tau},\delta_{s,\beta})$ plane. The
vertical angular scale is expanded by a factor of five to show the transit
loop, following the unequal-scale presentation in
[Mao & Witt (1998), Fig. 2](https://arxiv.org/pdf/astro-ph/9804045).
The cross marks zero displacement relative to the unlensed source. The
tracks include the wings out to $|t-t_0|/t_E=1000$, where they approach
the origin. Arrows show increasing time.
```

The two-dimensional figure plots the lensing displacement, not the
source's proper motion. The point-source ellipse and finite-source loop
are easier to compare separately from the time series.

### Isolating limb darkening

To resolve the smaller change between the two extended-source profiles,
subtract the uniform-disk prediction at the same epochs:

```{figure} ../example/astrometry/outputs/limb_darkening.png
:alt: Limb-darkened minus uniform-source residuals in magnification and two centroid components on a common time axis.

Limb-darkening residuals: $100(A_{\mathrm{linear}}/A_{\mathrm{uniform}}-1)$
in the top panel, and $\delta_{\mathrm{linear}}-\delta_{\mathrm{uniform}}$
in $\mu$as in the two astrometric panels. All panels share the same time
range and limb-contact markers as the preceding light curve. These are
differences between physical source profiles, rather than numerical errors.
```

Download the {download}`time-series PDF
<../example/astrometry/outputs/source_profiles.pdf>`,
{download}`centroid-track PDF <../example/astrometry/outputs/source_tracks.pdf>`,
or {download}`limb-darkening residual PDF
<../example/astrometry/outputs/limb_darkening.pdf>`.

## Blending and the observed centroid

Photometric blending needs only a source flux $F_s$ and a blend flux $F_b$:

$$
F(t)=F_s A(t)+F_b.
$$

Astrometric blending also needs the blend position
$\boldsymbol\theta_b(t)$ in the same coordinate frame and angular units as
the source. For constant intrinsic fluxes, an unmagnified blend, and
unresolved light, the measured centroid is

$$
\boldsymbol\theta_{\mathrm{obs}}(t)=
\frac{F_s A(t)[\boldsymbol\theta_s(t)+\boldsymbol\delta_s(t)]
      +F_b\boldsymbol\theta_b(t)}{F_s A(t)+F_b}.
$$

Define the unlensed combined centroid **at the same time** as

$$
\boldsymbol\theta_{\mathrm{base}}(t)=
\frac{F_s\boldsymbol\theta_s(t)+F_b\boldsymbol\theta_b(t)}{F_s+F_b}.
$$

Subtracting gives

$$
\begin{aligned}
\boldsymbol\delta_{\mathrm{obs}}(t)
&=\boldsymbol\theta_{\mathrm{obs}}(t)
 -\boldsymbol\theta_{\mathrm{base}}(t)\\
&=\frac{A}{A+g}\boldsymbol\delta_s
 +\frac{g(A-1)}{(A+g)(1+g)}
  (\boldsymbol\theta_s-\boldsymbol\theta_b),
\qquad g=\frac{F_b}{F_s}.
\end{aligned}
$$

The first term weights the source's lensing shift by its fraction of the
observed light. The second term comes from changing the source-to-blend
flux ratio as the source brightens. It vanishes when the source and blend
are co-located. In general, multiplying the unblended shift by a dilution
factor alone does not predict the observed shift: blending can increase a
component or change its direction, depending on the geometry.

The following helper implements the full centroid and its baseline using
JAX arrays and complex positions. It is example code, rather than a
`singlens` API function. Fluxes must describe the same passband as the
centroid measurement, with physical $F_s>0$ and $F_b\geq0$.
A negative nuisance `f_b` from an unconstrained photometric fit is not a
physical stellar blend flux for this model.

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN BLENDING MODEL
:end-before: END BLENDING MODEL
```

### A luminous lens and an unrelated blend

For a luminous lens, set $\boldsymbol\theta_b=\boldsymbol\theta_L$ and
$F_b=F_L$. The source-only lensing calculation is unchanged; the lens
light enters the final flux-weighted centroid. This treatment assumes a
point lens and does not include occultation by a lens of finite angular
radius. Luminous-lens centroids are discussed by
[Lee et al. (2010)](https://arxiv.org/abs/1005.3021).

For an unrelated unmagnified star, supply its own position and motion.
Its effect depends on its location as well as its brightness. In this
example the lens is fixed at the origin, the source position is
$\theta_s=\theta_E w(t)$, and the unrelated star is at $(3,2)$ mas. Both
blend scenarios use $F_b/F_s=0.5$ and the same limb-darkened source:

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN BLENDING EXAMPLE
:end-before: END BLENDING EXAMPLE
:dedent: 4
```

```{figure} ../example/astrometry/outputs/blending.png
:alt: Identical blended light curves but different astrometric shifts for a luminous lens and an unrelated star with the same blend flux.

Normalized flux and the two centroid-shift components share a single time
axis. The luminous lens and unrelated blend produce identical
$F/(F_s+F_b)$, shown once as "Both blends", but different astrometric
signals. Each blended shift is measured from its own unlensed combined
centroid at the same epoch. The unblended comparison is measured from the
unlensed source center.
```

```{figure} ../example/astrometry/outputs/blending_tracks.png
:alt: Centroid-shift tracks of an unblended source, a luminous lens, and an unrelated blend, plotted with equal angular scales.

Blending changes the shape and direction of the two-dimensional
displacement. This figure uses equal angular scales on both axes and
includes the same extended time sampling as the source-only tracks.
The origin denotes each model's own same-time unlensed centroid.
```

Thus a photometric estimate of `f_b` alone cannot determine the astrometric
correction. The blend's position or a model for it must also be constrained.
For several unmagnified blends, replace $F_b$ and
$F_b\boldsymbol\theta_b$ with

$$
F_b=\sum_j F_{b,j},\qquad
F_b\boldsymbol\theta_b(t)=\sum_j F_{b,j}\boldsymbol\theta_{b,j}(t).
$$

Their effective position is their flux-weighted centroid, which can move
with time. Lens light and unrelated stars can be combined this way when
they all contribute to the measured unresolved centroid. If the imaging
measurement excludes or resolves some components, use the fluxes and
centroid model appropriate to that measurement.

Download the {download}`blending time-series PDF
<../example/astrometry/outputs/blending.pdf>` or
{download}`blending centroid-track PDF
<../example/astrometry/outputs/blending_tracks.pdf>`.

## From a lensing shift to sky astrometry

To model measured positions, choose a common tangent-plane coordinate
system for the source, lens, and blends. For example, a source position
model can include a reference position, proper motion, and annual
parallax:

$$
\boldsymbol\theta_s(t)=\boldsymbol\theta_{s,0}
+\boldsymbol\mu_s(t-t_{\mathrm{ref}})
+\pi_s\boldsymbol P(t),
$$

where $\boldsymbol P(t)$ contains the parallax factors in the chosen axes
and time units match the proper motion. Add the rotated lensing shift to
this position, then combine the source images and blends using the
centroid equation above. For a luminous lens, source and lens positions
must obey the same relative trajectory,
$\theta_s(t)-\theta_L(t)=\theta_E e^{i\phi}w(t)$.

`AnnualParallax` changes the relative trajectory $w(t)$ used by both
`magnification` and `astrometric_shift`. It does not supply the source's
absolute position, proper motion, or source parallax $\pi_s$. Choose the
sky rotation consistently with the fitted microlensing parallax vector.
For the complex sky convention **north $+\,i$ east**, the package's
trajectory convention gives

$$
\phi=\operatorname{atan2}(\pi_{E,E},\pi_{E,N}),\qquad
\delta_N+i\delta_E
=\theta_E e^{i\phi}\,
\operatorname{astrometric\_shift}(t).
$$

Here east is the tangent-plane displacement $\Delta\mathrm{RA}\cos(\mathrm{Dec})$.
For a nonzero microlensing parallax vector, this orientation is determined
by its components; it is not an additional independent angle. The angle
refers to the geocentric $\tau$ direction at `AnnualParallax.t_ref`.
Without parallax, the sky orientation must be supplied separately. See the
[API](api.md#annual-parallax) for the time convention and geocentric
reference epoch.

The instantaneous baseline above isolates the displacement caused by
lensing. Subtracting a fixed reference-epoch position instead also retains
the unlensed source/blend motion between epochs. For an absolute-position
fit, use `observed_centroid` directly and fit those motions; for a
residual-position fit, reproduce the reference model used to construct
the observations.

For a joint fit, evaluate $A$ and $\delta_s$ with the same event and source
profile, compare `total_flux` with the photometry, and compare
`observed_centroid` (or the appropriately referenced `relative_shift`)
with the astrometry. Fit the source and blend fluxes, $\theta_E$, and
position/motion parameters as required by the data. Fluxes and source
brightness profiles can differ between passbands while the lensing
geometry is shared.

## Accuracy and reproducibility

Use float64. Astrometric precision must be checked in angular units;
relative errors become uninformative when a signed shift crosses zero.
Photometric accuracy does not by itself bound the centroid error. Finite
source contact, approximation transitions, and derivatives require
particular attention; see
[astrometric numerical accuracy](numerics.md#astrometric-shifts) for the
validation results and resolution checks.

For blended astrometry, propagate numerical changes in both $A$ and
$\delta_s$ through the full centroid model. Magnification errors also
change the weighting of spatially separated components, so the final
angular precision depends on the blend geometry.

The {download}`complete executable example
<../example/astrometry/code/astrometry.py>` generates all five figures. From
the repository root, with the `examples` extra installed, run:

```bash
python example/astrometry/code/astrometry.py
```

PNG and PDF figures are written to `example/astrometry/outputs/`.
Use `--output-dir PATH` to choose another location. For prepared models,
JIT compilation, automatic differentiation, and evaluation at supplied
source positions, see the {ref}`astrometry API <astrometric-shift-api>`.
