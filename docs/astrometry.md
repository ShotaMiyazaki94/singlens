# Astrometric microlensing

`model.joint(times, params)` returns magnification and the centroid shift of
unresolved source images. An observed sky centroid adds angular scaling,
sky orientation, source motion, and blended light. See
[finite-source effects](finite-source-effects.md) for solver accuracy.
For parameter estimation, [Photometry + Astrometry](roman-astrometry.md)
shows the joint likelihood, NumPyro model, and posterior recovery.

(what-the-shift-represents)=
## Coordinates and physical scales

The source-only shift is the displacement of the lensed-image centroid from
the unlensed source center:

$$
\boldsymbol\delta_s(t)
=\boldsymbol\theta_{\mathrm{img}}(t)-\boldsymbol\theta_s(t).
$$

`model.astrometric_shift(times, params)` returns $\boldsymbol\delta_s/\theta_E$ as a complex
array in the trajectory's fixed $\tau,\beta$ axes. For a static event,

$$
w(t)=\frac{t-t_0}{t_E}+iu_0,\qquad
\frac{\delta_s(t)}{\theta_E}=\frac{w(t)}{|w(t)|^2+2}
\quad\text{(point source)}.
$$

Here $w$ is the source-minus-lens separation in Einstein-radius units.
Multiply the relative shift by $\theta_E$ in mas for a displacement in mas;
`.real` and `.imag` give its signed components. A factor $e^{i\phi}$ rotates
the trajectory axes into a sky frame.

In the movies, the cross denotes **zero lensing displacement from the
unlensed reference at the same epoch**. The first two use the source center;
the luminous-lens movie uses each curve's combined-light centroid. The
$\tau,\beta$ axes stay fixed even when annual parallax bends the trajectory.

For an extended source, the centroid is the image's first position moment
divided by its magnified flux. This weighting can reverse the shift during
a source transit and deform the point-source ellipse
([Mao & Witt 1998](https://arxiv.org/abs/astro-ph/9804045),
[Lee et al. 2010](https://arxiv.org/abs/1005.3021)). The
[theory](theory/index.md#astrometric-centroid-shift) gives the moment integrals.

(physical-scales-of-the-examples)=
### Physical scales

The three examples describe a planetary-mass lens crossing a giant, an
isolated black hole, and a luminous G-dwarf lens. Their scales satisfy

$$
\pi_{\rm rel}=\mathrm{AU}\left(\frac{1}{D_L}-\frac{1}{D_S}\right),\qquad
\theta_E^2=\kappa M_L\pi_{\rm rel},\qquad
\pi_E=\frac{\pi_{\rm rel}}{\theta_E},\qquad
\mu_{\rm rel}=\frac{\theta_E}{t_E},\qquad
\theta_*=\rho\theta_E,
$$

with $\kappa\simeq8.144\ \mathrm{mas}\,M_\odot^{-1}$ and
$\pi_E=|\boldsymbol\pi_E|$ ([Gould 2000](https://arxiv.org/abs/astro-ph/0001421)).
Distances in kpc give $\pi_{\rm rel}$ in mas as $1/D_L-1/D_S$.
Proper motion and $t_E$ must share a reference frame; the parallax examples
use geocentric motion at `t_ref`.

For the illustrative source transit, $D_S=8$ kpc, $\pi_{\rm rel}=0.1$ mas,
$\theta_E=0.012$ mas, $t_E=0.8$ days, and $\rho=0.5$. These imply
$M_L\simeq0.185\,M_{\rm Jup}$, $D_L\simeq4.44$ kpc,
$\mu_{\rm rel}\simeq5.48$ mas yr$^{-1}$, and a giant with
$\theta_*=6\,\mu$as ($R_*\simeq10.3\,R_\odot$). For comparison,
OGLE-2016-BLG-1540 has measured $t_E=0.32$ days and
$\theta_E=9.2\,\mu$as ([Mróz et al. 2018](https://arxiv.org/abs/1712.01042)).

## Light curves and centroid tracks

(light-curve-and-centroid-of-the-same-event)=
### A source transit

With $u_0=0.1<\rho=0.5$, the lens crosses the giant's disk. Compare the
same trajectory for a point source, a uniform disk, and a linear
limb-darkened disk with illustrative `a1=0.6`. Each movie pairs photometry
and centroid tracks with synchronized time markers; the lens is a point
mass, without finite-lens occultation.

```python
from typing import NamedTuple

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import singlens as sl
```

Prepare a `Model` for each source profile and use the same event parameters
for photometry and astrometry:

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN SOURCE EXAMPLE
:end-before: END SOURCE EXAMPLE
:dedent: 4
```

The [linear coefficient](theory/index.md#finite-sources-and-limb-darkening)
depends on stellar atmosphere and passband. Point-source `Parameters` omit
`rho` and `a1`; uniform-disk parameters include `rho` and omit `a1`.

```{figure} ../example/astrometry/outputs/source_animation.gif
:alt: Synchronized light curves and centroid-shift tracks for point, uniform, and limb-darkened sources during a transit.

Photometry (left) and source-relative centroid shifts (right), from
$t_0-3.2$ to $t_0+3.2$ days at constant playback speed. Dotted lines mark
limb contact at $t-t_0\simeq\pm0.392$ days. The vertical angular scale is
expanded fivefold to show the transit loop, following
[Mao & Witt (1998), Fig. 2](https://arxiv.org/pdf/astro-ph/9804045).
Faint tracks extend to $|t-t_0|/t_E=1000$; highlighted tracks cover the movie.
The cross is zero displacement from the unlensed source.
```

A uniform disk has zero shift at $u=\rho$ and at alignment; limb darkening
changes the contact behavior. These tracks omit source proper motion and
annual parallax. The faint extensions show ideal rectilinear limits rather
than a multi-year observing prediction.

Download the {download}`animation
<../example/astrometry/outputs/source_animation.gif>`.

(a-long-event-with-annual-parallax)=
### Annual parallax in a long event

For **MOA-2011-BLG-191 / OGLE-2011-BLG-0462**, adopt the complete geocentric
photometric/astrometric solution in
[Mróz et al. (2022), Table 2, “Sahu+”](https://arxiv.org/html/2207.10729v2#S5.T2):
$t_E=239.7$ days, $\theta_E=5.68$ mas, and $D_S=5.9$ kpc imply a
$7.88\,M_\odot$ lens at $1.49$ kpc. The model uses a point source and a
dark lens; the event coordinates are
$(\mathrm{RA},\mathrm{Dec})=(267.9175342^\circ,-29.8906950^\circ)$
([Sahu et al. 2022, Table 1](https://arxiv.org/html/2201.13296#S3.T1)).

Compare annual parallax with rectilinear motion at the same parameters.
Times and `t_ref` are HJD $-2450000$. From the repository root, import:

```python
from example.astrometry.code.long_event import calculate_parallax_comparison
```

The helper evaluates both trajectories at matching epochs:

```{literalinclude} ../example/astrometry/code/long_event.py
:language: python
:start-after: BEGIN LONG EVENT EXAMPLE
:end-before: END LONG EVENT EXAMPLE
:dedent: 4
```

```{figure} ../example/astrometry/outputs/long_event_animation.gif
:alt: Synchronized light curves and centroid tracks comparing annual parallax with rectilinear motion in a long event.

Annual parallax bends the centroid track over $t_0\pm719.1$ days
($\pm3t_E$), shown at constant playback speed. Logarithmic magnification
reveals the narrow peak ($A_{\max}\simeq380$) and broad wings. Angular axes
have equal scales in the fixed $\tau,\beta$ frame; the cross is zero shift
from the same-epoch unlensed source.
```

These curves show source magnification and lensing displacement, excluding
OGLE blending and unlensed source motion. The published photometry-only and
heliocentric solutions are separate parameter sets.

Download the {download}`long-event animation
<../example/astrometry/outputs/long_event_animation.gif>`.

(blending-and-the-observed-centroid)=
## Blending and sky astrometry

### Observed centroid

Photometry predicts $F(t)=F_s A(t)+F_b$. Astrometry also requires the blend
position $\boldsymbol\theta_b(t)$ in the source's coordinate frame and
angular units. For constant intrinsic fluxes and an unresolved,
unmagnified blend,

$$
\boldsymbol\theta_{\mathrm{obs}}(t)=
\frac{F_s A(t)[\boldsymbol\theta_s(t)+\boldsymbol\delta_s(t)]
      +F_b\boldsymbol\theta_b(t)}{F_s A(t)+F_b}.
$$

Subtract the unlensed combined centroid **at the same epoch**,

$$
\boldsymbol\theta_{\mathrm{base}}(t)=
\frac{F_s\boldsymbol\theta_s(t)+F_b\boldsymbol\theta_b(t)}{F_s+F_b},
$$

to obtain

$$
\boldsymbol\delta_{\mathrm{obs}}
=\frac{A}{A+g}\boldsymbol\delta_s
 +\frac{g(A-1)}{(A+g)(1+g)}
  (\boldsymbol\theta_s-\boldsymbol\theta_b),
\qquad g=\frac{F_b}{F_s}.
$$

The second term accounts for changing light weights and vanishes for
co-located source and blend. Dilution alone misses this term, which can
change the shift's size or direction.

Use same-passband physical fluxes $F_s>0$, $F_b\geq0$; a negative nuisance
`f_b` from a photometric fit cannot represent stellar blend light.

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN BLENDING MODEL
:end-before: END BLENDING MODEL
```

The helper returns total flux, observed centroid, unlensed baseline, and relative shift.

(a-g-dwarf-lens-on-roman-event-scales)=
### A luminous G-dwarf lens

For lens light, set $\boldsymbol\theta_b=\boldsymbol\theta_L$ and $F_b=F_L$.
Lens light contributes to the flux-weighted observed centroid
([Lee et al. 2010](https://arxiv.org/abs/1005.3021)).
This example has a solar-mass G-dwarf lens at 6 kpc and a solar-like source
at 8 kpc, with $t_E\simeq42.6$ days and small finite-source effects.
Dwarf sources occur in Roman yield calculations
([Penny et al. 2019, Fig. 1](https://arxiv.org/pdf/1808.02490)).
Equal intrinsic luminosity and foreground extinction give $F_L/F_s=1.78$
in [F146](https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey).

Use the lens-relative frame, $\theta_L=0$ and $\theta_s=\theta_E w(t)$,
with the same annual-parallax trajectory in both observables:

```python
from example.astrometry.code.long_event import calculate_roman_event

roman_event = calculate_roman_event()
```

```{literalinclude} ../example/astrometry/code/astrometry.py
:language: python
:start-after: BEGIN BLENDING EXAMPLE
:end-before: END BLENDING EXAMPLE
:dedent: 4
```

```{figure} ../example/astrometry/outputs/blending_animation.gif
:alt: Synchronized normalized flux and centroid tracks comparing dark and luminous G-dwarf lenses.

Normalized flux and centroid shifts with annual parallax, over
$t_0\pm170.21$ days ($\pm4t_E$) at constant playback speed. Each light
curve uses its own baseline flux. The cross is zero shift from each
curve's same-epoch unlensed centroid: source-only for the dark lens,
source-plus-lens for the luminous lens. Overlapping tracks would therefore
not imply equal sky positions. Angular axes have equal scales.
```

`AnnualParallax` uses Earth's orbit. A Roman observing model additionally
needs the L2 ephemeris, seasons, cadence, and errors. The
[joint inference example](roman-astrometry.md) adds noisy, seasonal data
and fits the unknown baseline position, proper motion, and parallax.

For several unrelated blends, use their total flux and flux-weighted position,
including each component's motion and parallax.

Download the {download}`blending animation
<../example/astrometry/outputs/blending_animation.gif>`.

(from-a-lensing-shift-to-sky-astrometry)=
### Sky coordinates

In a common tangent plane, an unlensed source can follow

$$
\boldsymbol\theta_s(t)=\boldsymbol\theta_{s,0}
+\boldsymbol\mu_s(t-t_{\mathrm{ref}})
+\pi_s\boldsymbol P(t).
$$

Use matching time units for proper motion and parallax factors
$\boldsymbol P(t)$. Add the rotated lensing shift, then combine source and
blend light. A luminous lens must obey the same relative trajectory:
$\theta_s-\theta_L=\theta_E e^{i\phi}w(t)$.
`AnnualParallax` supplies $w(t)$, not the absolute source position, proper
motion, or source parallax $\pi_s$.

For complex sky coordinates **north $+\,i$ east**, the package convention is

$$
\phi=\operatorname{atan2}(\pi_{E,E},\pi_{E,N})+\pi,\qquad
\delta_N+i\delta_E
=\theta_E e^{i\phi}\,
\operatorname{astrometric\_shift}(t).
$$

East means $\Delta\mathrm{RA}\cos(\mathrm{Dec})$. The extra $\pi$ converts
the parallax vector's **lens-minus-source** proper-motion direction to
$w$'s **source-minus-lens** direction
([Park et al. 2004](https://www.astronomy.ohio-state.edu/gaudi.1/moa03037.pdf)).
For nonzero parallax this fixes the orientation of the geocentric $\tau$
axis at `AnnualParallax.t_ref`; without parallax, supply an orientation.
See the [API](api.md#annual-parallax) for time and reference-epoch conventions.

For absolute-position data, fit `observed_centroid` including baseline
motion. For residual positions, subtract the same reference model used
for the observations. A fixed reference-epoch position retains subsequent
unlensed motion; the same-epoch baseline isolates lensing displacement.
A joint fit compares `total_flux` with photometry and `observed_centroid` with
positions. Share the lensing geometry while fitting fluxes, $\theta_E$, and
baseline motion; brightness profiles may differ between passbands.

(accuracy-and-reproducibility)=
## Reproduce and validate

Use float64 and assess errors in angular units, including changes in both
magnification and shift when blending alters the light weights. See
[finite-source effects](finite-source-effects.md) for measured solver errors.

The {download}`executable example
<../example/astrometry/code/astrometry.py>` uses the
{download}`long-event model <../example/astrometry/code/long_event.py>` and
{download}`animation helper <../example/astrometry/code/animation.py>`.
With the `examples` extra installed, run from the repository root:

```bash
python example/astrometry/code/astrometry.py
```

The three GIFs, CSV data, and event configurations go to
`example/astrometry/outputs/`; `--output-dir PATH` selects another location.
See the {ref}`astrometry API <astrometric-shift-api>` for evaluation and differentiation.
