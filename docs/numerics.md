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

## Annual parallax

The bundled Earth ephemeris covers 2000–2050 with daily sampling. Time
derivatives are limited by that sampling. Out-of-range queries, including
those shifted outside the table by HJD conversion, return NaN.

Keep the geocentric reference epoch fixed while fitting `t0`. See
[the API](api.md#annual-parallax) for time conventions and
[theory](theory/index.md) for trajectory and limb-darkening definitions.
