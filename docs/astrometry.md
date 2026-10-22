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

Each example pairs a light curve on the left with a centroid-shift track
on the right. The time cursor and moving markers stay synchronized.
The curves are new `singlens` point-lens calculations; finite-lens
occultation is outside these examples' model.

```python
from typing import NamedTuple

import jax
import numpy as np

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

```{figure} ../example/astrometry/outputs/source_animation.gif
:alt: Synchronized animation of the three light curves on the left and their two-dimensional centroid-shift tracks on the right, with moving markers at the same time.

The light curves (left) and centroid-shift tracks (right) evolve together
from $t_0-80$ to $t_0+80$ days at a constant playback rate. The time cursor
and moving markers identify the same epoch in both panels. Faint tracks
on the right show the full paths out to $|t-t_0|/t_E=1000$; the highlighted
paths trace the animation's time interval. Dotted lines on the left mark
limb contact at $t-t_0\simeq\pm9.80$ days. The vertical angular scale is
expanded by a factor of five to show the transit loop, following
[Mao & Witt (1998), Fig. 2](https://arxiv.org/pdf/astro-ph/9804045).
The cross marks zero displacement relative to the unlensed source.
```

The right panel plots the lensing displacement relative to the source;
the source's proper motion is not included in these tracks. Finite source
size changes both the peak light curve and the centroid track. A uniform
disk has zero shift at $u=\rho$ as well as at exact alignment; the behavior
near contact depends on the brightness profile.

Download the {download}`animation
<../example/astrometry/outputs/source_animation.gif>`.

## A long event with annual parallax

For a longer event, Earth's orbital motion changes both the light curve
and the astrometric trajectory. This example uses $t_E=180$ days,
$u_0=0.35$, $\rho=0.005$, and $\theta_E=1.2$ mas, with
$(\pi_{E,N},\pi_{E,E})=(0.12,0.08)$. The source is a uniform disk and
remains well separated from the lens compared with its radius, so the
finite-source effect is small.

The sky position is $(\mathrm{RA},\mathrm{Dec})=(266^\circ,-29^\circ)$,
and $t_0=t_{\mathrm{ref}}=8000$ in HJD $-\,2450000$. Both models use the
same event and source parameters; one includes `AnnualParallax` and the
other follows rectilinear motion:

```{literalinclude} ../example/astrometry/code/long_event.py
:language: python
:start-after: BEGIN LONG EVENT EXAMPLE
:end-before: END LONG EVENT EXAMPLE
:dedent: 4
```

```{figure} ../example/astrometry/outputs/long_event_animation.gif
:alt: Synchronized light curves and astrometric tracks for a long event, comparing annual parallax with rectilinear motion.

The same epochs are highlighted in the light curves and centroid tracks,
from $t_0-540$ to $t_0+540$ days at a constant playback rate. Annual
parallax produces an asymmetric light curve and a non-elliptical centroid
track. Both angular axes use the same scale. The shifts are relative to
the unlensed source, in the trajectory's $\tau,\beta$ frame.
```

Download the {download}`long-event animation
<../example/astrometry/outputs/long_event_animation.gif>`.

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

### A luminous lens

For a luminous lens, set $\boldsymbol\theta_b=\boldsymbol\theta_L$ and
$F_b=F_L$. The source-only lensing calculation is unchanged; the lens
light enters the final flux-weighted centroid. This treatment assumes a
point lens and does not include occultation by a lens of finite angular
radius. Luminous-lens centroids are discussed by
[Lee et al. (2010)](https://arxiv.org/abs/1005.3021).

The following comparison uses the long event above, including annual
parallax, with a lens-to-source flux ratio $F_L/F_s=0.5$. Work in the
lens-relative frame, with $\theta_L=0$ and
$\theta_s=\theta_E w(t)$; the relative trajectory $w(t)$ includes the same
parallax in the photometric and astrometric calculations:

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN BLENDING EXAMPLE
:end-before: END BLENDING EXAMPLE
:dedent: 4
```

```{figure} ../example/astrometry/outputs/blending_animation.gif
:alt: Synchronized normalized light curves and centroid-shift tracks for the long parallax event, comparing no lens light with a luminous lens.

Normalized flux (left) and centroid-shift geometry (right) for the same
long event, including annual parallax. The luminous-lens shift is measured
from the unlensed source-plus-lens centroid at the same epoch. The unblended
comparison is measured from the
unlensed source center; each light curve is normalized by its own baseline
flux. Both angular axes use the same scale. The movie covers $t_0\pm540$
days, with the same playback rate as the preceding example.
```

Unrelated unresolved light can be included when required by an individual
analysis: supply its flux (assumed constant here) and position as a function
of time, including proper motion and parallax as needed. These are
additional observation-model parameters; a photometric blend flux alone
does not specify them. Several such components can be combined through
their total flux and flux-weighted position.

Download the {download}`blending animation
<../example/astrometry/outputs/blending_animation.gif>`.

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
<../example/astrometry/code/astrometry.py>` generates the three GIFs,
using the {download}`long-event model
<../example/astrometry/code/long_event.py>` and
{download}`animation helper <../example/astrometry/code/animation.py>`.
From the repository root, with the `examples` extra installed, run:

```bash
python example/astrometry/code/astrometry.py
```

GIFs, numerical CSV data, and event configurations are written to
`example/astrometry/outputs/`.
Use `--output-dir PATH` to choose another location. For prepared models,
JIT compilation, automatic differentiation, and evaluation at supplied
source positions, see the {ref}`astrometry API <astrometric-shift-api>`.
