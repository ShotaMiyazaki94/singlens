# Changelog

## 0.1.0

- Event-parameter magnification API with annual parallax, finite sources,
  dynamic linear limb darkening, JIT, and automatic differentiation.
- Default 4096-bin finite-source settings checked at about 0.1% relative error
  down to `rho=1e-6` against independent source integration.
- Roman-like injection–recovery with joint parallax, limb darkening, flux,
  and tinygp source-rotation inference; magnitude residuals, corner, and traces.
- Magnification accuracy/runtime comparisons with VBMicrolensing and
  MulensModel, plus VBM/emcee versus singlens/NUTS inference on independent-noise
  data, with overlaid posteriors, traces, and a common ESS target.
- Named event parameters, source-position evaluation,
  flux fitting, and a bundled Earth ephemeris covering 2000–2050.

Requires Python 3.12+, JAX 0.11.2+, and float64 for finite-source calculations.
