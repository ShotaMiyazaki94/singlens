# Numerical accuracy

Use float64 for finite-source calculations. The default FFT resolution is
4096 bins. [Benchmarks](benchmarks.md) report accuracy against independent
source integration and comparisons with other microlensing packages.

## FFT resolution

Use **4096 bins** by default. Independent integration checks cover 31 source
radii from `1e-6` to 10, near alignment, source contact, and far wings.
The largest sampled relative error is **0.101%**, for a uniform source near
contact. Uniform brightness and pure linear limb darkening bracket physical
linear coefficients `0 <= a1 <= 1`; the pure quadratic profile is checked separately.

At `rho=.01`, the largest sampled errors are
**0.093% for a uniform source** and **0.062% for linear limb darkening**
(`a1=.5`). Higher resolution can check a particular curve, derivative, or
posterior. These are sampled results, not error bounds for arbitrary events.
Assess numerical differences relative to the measurement errors when fitting
precise data.

The default transform grid spans `u=1e-8` to `1e4`. This does not require
observations at such large separations. The
[original method](https://arxiv.org/html/2203.06637v1#S3) explains why small
sources need high-frequency modes, and uses roughly `1e-6` to `1e3` with
1024 bins in its demonstration. That range is not presented as an optimum.

Keep the default range unless you have checked the intended parameter domain.
Changing it changes frequency coverage and endpoint effects; narrowing it
does not reduce FFT work at a fixed bin count. Observation count, FFT
resolution, and transform range are separate settings.

## Derivatives

Automatic differentiation evaluates derivatives of the same numerical model
used for magnification. Check their stability near source contact and exact
alignment when they determine a fit.

With the defaults, approximations change at `rho=1e-4` and, for smaller
sources, at `u/rho=20`. Near alignment (`u/rho<.01`), a central expansion
preserves the value and leading slope. Derivatives need not be continuous
at approximation boundaries. At `u>=100` in the larger-source calculation,
magnification is fixed to 1. More FFT bins do not remove these limits.

## Annual parallax

The bundled Earth ephemeris covers 2000–2050 with daily sampling. Time
derivatives are limited by that sampling. Out-of-range queries, including
those shifted outside the table by HJD conversion, return NaN.

Keep the geocentric reference epoch fixed while fitting `t0`. See
[the API](api.md#annual-parallax) for time conventions and
[theory](theory/index.md) for trajectory and limb-darkening definitions.
