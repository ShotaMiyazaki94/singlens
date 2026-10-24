# Numerical accuracy

Use float64 for finite-source calculations. [Benchmarks](benchmarks.md) report
magnification accuracy against independent source integration and comparisons
with other microlensing packages. Those errors do not establish astrometric
accuracy; see [astrometric shifts](#astrometric-shifts) below.

## FFT resolution

Use **4096 bins** by default. Independent integration checks cover 31 source
radii from `1e-6` to 10, near alignment, source contact, and far wings.
The largest sampled relative error is **0.101%**, for a uniform source near
contact. Uniform brightness and pure linear limb darkening bracket physical
linear coefficients `0 <= a1 <= 1`; the pure quadratic profile is checked separately.

These are sampled results, not error bounds for arbitrary events.
Higher resolution can check a particular curve, derivative, or posterior.
Assess numerical differences relative to the measurement errors when fitting
precise data.

For a particular limb-darkened curve, compare resolutions at identical
separations and parameters. The following also compares radius derivatives;
the input positions remain fixed while `rho` is differentiated:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import singlens as sl

rho = 0.01
u = rho * jnp.linspace(0.0, 3.0, 301)
values, derivatives = {}, {}
for bins in (2048, 4096, 8192):
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=bins))

    def curve(radius):
        return model.magnification_at(u, rho=radius, a1=0.5)

    values[bins] = np.asarray(jax.jit(curve)(rho))
    derivatives[bins] = np.asarray(jax.jit(jax.jacfwd(curve))(rho))

relative_change = np.max(np.abs(values[4096] / values[8192] - 1))
slope_scale = np.max(np.abs(derivatives[8192]))
slope_change = np.max(np.abs(derivatives[4096] - derivatives[8192])) / slope_scale
print("Maximum relative magnification change:", relative_change)
print("Derivative change / peak derivative:", slope_change)
```

This measures a change in the approximation, not an error against an exact
answer. Compare to independent source integration when an accuracy reference
is needed; [benchmarks](benchmarks.md#accuracy-versus-fft-bins) show that calculation.
For a fit, `f_s * (A_4096 - A_8192) / flux_error` compares the curve difference
to each observation's flux uncertainty. Small per-point differences can still
accumulate in a likelihood, so examine posterior stability as well.

Derivative differences are scaled by the largest reference derivative above;
pointwise relative differences are unstable where a derivative crosses zero.
Magnification error measurements do not guarantee the same accuracy for
derivatives, especially near source contact.

The FFT grid spans `u=1e-8` to `1e4`. This is the calculation's integration
range; observations do not need to cover it. Changing the range affects
numerical accuracy, while narrowing it leaves the FFT size and its cost
unchanged at a fixed bin count. Keep the default unless accuracy has been
checked for the source radii you need. The
[original method](https://arxiv.org/html/2203.06637v1#S3) describes the
frequency coverage needed for small sources.

## Distant observations

With the defaults, positions at **`u/rho >= 20`** use the point-source
magnification plus the leading finite-source correction. The correction
depends on both source radius and radial brightness profile, so radius and
limb-darkening derivatives are retained. This applies to every source radius.

**When every position in one event evaluation satisfies this condition,
the radius-dependent FFT is skipped.** A mixed batch uses the same expression
at its distant positions. Between `u/rho=10` and 20, a smooth blend connects
the finite-source calculation to the far-field expansion.

Closer positions use the FFT calculation for `rho >= 1e-4`; smaller sources
use a small-source approximation. `FFTConfig.u_switch` sets the outer transition
ratio, with the blend starting at half that value.

## Derivatives

Automatic differentiation evaluates derivatives of the same numerical model
used for magnification. Check their stability near source contact and exact
alignment when they determine a fit.

With the defaults, approximations change at `rho=1e-4`.
Near alignment (`u/rho<.01`), a central expansion
preserves the value and leading slope. Derivatives need not be continuous
at these approximation boundaries. The transition to the far-field
expansion is smooth. More FFT bins do not remove the other branch limits.

## Astrometric shifts

Finite-source astrometry combines the lensed flux and an image-position first
moment. Errors in these quantities can cancel or reinforce in the centroid,
so the magnification percentages above do not bound the position error.
Check the flux, moment, and final shift against independent source integration
when establishing accuracy for a new parameter range.

Use **absolute angular errors**, scaled by the event's angular Einstein radius.
Pointwise relative errors are undefined at exact alignment and at other
zeroes of the astrometric shift. For example, compare resolutions at fixed
source positions and differentiate at those same positions:

```python
rho = 0.01
position = rho * (jnp.linspace(-3.0, 3.0, 301) + 0.2j)
shifts, slopes = {}, {}
for bins in (4096, 8192):
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=bins))

    def shift_at_radius(radius):
        return model.astrometric_shift_at(position, rho=radius, a1=0.5)

    shifts[bins] = np.asarray(jax.jit(shift_at_radius)(rho))
    slopes[bins] = np.asarray(jax.jit(jax.jacfwd(shift_at_radius))(rho))

theta_e_mas = 1.2
shift_change_mas = theta_e_mas * np.max(np.abs(shifts[4096] - shifts[8192]))
slope_change = np.max(np.abs(slopes[4096] - slopes[8192])) / np.max(np.abs(slopes[8192]))
print("Maximum shift change [mas]:", shift_change_mas)
print("Derivative change / peak derivative:", slope_change)
```

This is a convergence comparison, not an independent error estimate. Compare
angular differences with the astrometric measurement uncertainties and check
posterior stability when accumulating many epochs. Test gradients separately,
particularly near source contact and approximation transitions.
For a uniform disk, the sharp brightness edge makes the immediate vicinity
of contact sensitive to the high-frequency regularization as well as the
grid. Increasing the bin count alone does not remove that smoothing bias;
use independent integration to assess it for the required angular precision.

The default float64 implementation was compared with independent image-moment
integration at **1,458 sampled values**, covering eight source radii from
`1e-6` to 10, uniform disks, pure linear limb darkening (`a1=1`), and the pure
`mu**2` profile. The samples include 41 separations between `u/rho=0.99` and
1.01, approximation boundaries, alignment, and distant observations. The
largest absolute error was **`1.241e-3 * theta_E`**, for a uniform disk with
`rho=10` at `u/rho=1.002`. The pure linear and `mu**2` profile maxima were
`1.581e-4 * theta_E` and `1.956e-5 * theta_E`, respectively. These measurements
describe the sampled domain; they are not a uniform error bound. Reproduce
them with `python -m tools.validation.astrometry_accuracy`; its
`1.5e-3 * min(rho, 1)` threshold is a regression guard in Einstein-radius units.

Derivative accuracy needs a separate resolution check. For example, with a
uniform disk at `rho=0.1` and `u/rho=0.5`, the default 4096-point grid's radius
derivative differs from finite differences of independent integration by about
**2%**. At 8192 points, the radius and position derivatives at that point both
agree to better than `1e-7` in relative terms. The independent gradient tests
therefore use 8192 points; separate tests check that automatic differentiation
of the default model agrees with finite differences of that same numerical
model. Agreement in this latter check does not establish physical accuracy.

Useful analytic checks are the point-source limit
`position / (abs(position)**2 + 2)`, zero shift at alignment, and, for a uniform
disk, zero shift at `u=rho` and the central slope `-1/(rho**2+4)`.
The point-source Cartesian slope at alignment is `1/2`; taking zero source
radius and approaching exact alignment are different limits. Check rotational
covariance and the signed shift through a source transit as well.

Astrometric calculations retain the distant `1/u` displacement and its leading
finite-source correction. As with magnification, an event containing only
distant positions skips the radius-dependent FFT, with a smooth transition
set by `FFTConfig.u_switch`. Small-source and central approximations are
validated separately from FFT resolution. In particular, increasing the bin
count does not test the change of approximation at `FFTConfig.rho_switch`.

## Annual parallax

The bundled Earth ephemeris covers 2000–2050 with daily sampling. Time
derivatives are limited by that sampling. Out-of-range queries, including
those shifted outside the table by HJD conversion, return NaN.

Keep the geocentric reference epoch fixed while fitting `t0`. See
[the API](api.md#annual-parallax) for time conventions and
[theory](theory/index.md) for trajectory and limb-darkening definitions.
