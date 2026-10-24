# Changelog

## Unreleased

- Add `Model.marginalize_flux` and `singlens.photometry.marginalize_flux`
  for normalized Gaussian-prior integration of source and blend fluxes,
  conditional flux means/covariances, and differentiable event inference.
  Preserve prior uncertainty for constant curves with diffuse proper priors.

- Explicit low-level finite-source quadrature, Gould--Yoo approximations,
  and reusable cubic basis tables for photometry and astrometry.
- A bundled finite-source basis table, an offline generation tool, and
  numerical tests for values, gradients, precision changes, and table reuse.
  Model continues to use its existing FFTLog configuration.

- Astrometric centroid shifts for dark point lenses with point sources or
  circular uniform, linear limb-darkened, and pure `mu2` sources.
- `astrometric_shift`, `Model.astrometric_shift`, and direct position evaluation
  return source-relative shifts in Einstein-radius units, with annual
  microlensing parallax, JIT, automatic differentiation, and event batching.
- Separate finite-source flux and first-moment convolutions, analytic central
  and distant limits, and lazy preparation of the astrometric kernels.
- An astrometry chapter and reproducible light-curve/centroid examples,
  including luminous-lens and unrelated blending with an explicit
  flux-weighted observation model, shared-time-axis figures, separate
  centroid tracks, and limb-darkening residuals.

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
