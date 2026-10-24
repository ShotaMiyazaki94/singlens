# Changelog

## Unreleased

- **Breaking API change:** event evaluation requires `Model(source=...)` and
  `Parameters`; `source` has no default. Top-level `magnification`,
  `astrometric_shift`, and `fit_flux` are removed. For example, replace
  `sl.magnification(times, **event)` with:

  ```python
  model = sl.Model(source="linear")
  params = sl.Parameters(**event)  # Include rho and a1 for a linear source.
  A = model.magnification(times, params)
  fit = model.fit_flux(times, params, observed_flux, flux_error)
  ```

  Put `AnnualParallax` on the model and `pi_e_n`, `pi_e_e` in `Parameters`.
  Use `source="point"` explicitly for a point source. Analytic kernels and
  specialized source solvers are available from `singlens.lensing`; standalone
  flux utilities are in `singlens.photometry`. The package root exports only
  `__version__`, `Model`, `Parameters`, `AnnualParallax`, and `FluxFit`.
  The `lightcurve` module is removed.
- `Model.marginalize_flux` analytically integrates source and blend fluxes
  under explicit Gaussian priors, returning a normalized log likelihood
  and conditional flux mean/covariance for differentiable inference.
- Table evaluation reuses shared-radius spline coefficients across epochs,
  reducing work without changing the interpolation or public settings.
- Uniform and linear models default to a bundled 192 × 947 joint basis table;
  users no longer build tables for ordinary evaluation. Its finite domain is
  `1e-6 <= rho <= 10`, `0 <= u/rho <= 1e8`; outside values return NaN, while
  `rho=0` uses the analytic limit. Pure `mu2` models use FFTLog.
  Select fixed settings with `solver="table"` or `solver="fft"` (4096 bins).
  `Model.fft` and the root `FFTConfig`/`BasisTable` exports are removed.
  Ordinary FFT evaluation uses `Model(source="linear", solver="fft")`;
  numerical development can import `FFTConfig` and `BasisTable` from
  `singlens.lensing` and pass an object to `solver`. `ConcentricDisk` is an
  internal table-generation/reference implementation, not a public export
  or a `Model` solver option.
- Shared flux and first-moment concentric-disk quadrature, four-channel
  astrometric tables with analytic point-source baselines, and fixed
  `Model(source=..., solver=...)` backends with `joint`/`joint_at` evaluation. Independent
  moment and Cartesian-gradient checks, including alignment and distant
  wings, and reproducible simultaneous photometric/astrometric NUTS comparisons.
- Internal concentric-disk and specialized Gould–Yoo photometric solvers, higher-order
  small-source moments, and offline two-basis cubic interpolation, with
  independent value/gradient checks and reproducible runtime/NUTS comparisons.
- Astrometric centroid shifts for dark point lenses with point sources or
  circular uniform, linear limb-darkened, and pure `mu2` sources.
- `Model.astrometric_shift` and direct position evaluation
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
