# Photometry, astrometry, and blending

```bash
python example/astrometry/code/astrometry.py
```

Run from the repository root with the `examples` extra installed. The script
enables JAX 64-bit mode and writes PNG/PDF figures, CSV data, and the physical
configuration to `outputs/`. Override the destination with `--output-dir PATH`.

- `source_profiles`: magnification and both signed centroid-shift components
  for point, uniform, and linear sources, stacked on a shared time axis.
- `source_tracks`: a separate two-dimensional centroid plot, with the vertical
  angular scale expanded by a factor of five to show the finite-source loop.
- `limb_darkening`: linear-minus-uniform residuals in magnification and both
  shift components, stacked on the same time axis.
- `blending`: compare the unblended source, light from the lens, and light from
  an unrelated star in three panels with a shared time axis. The identical
  blended photometric curves are drawn once; the centroids differ.
- `blending_tracks`: separate two-dimensional blend trajectories with equal
  angular scales.

The event has `t0=0 day`, `u0=0.1`, `t_e=20 day`, `rho=0.5`, and
`theta_E=1.2 mas`; the linear profile uses `a1=0.6`. Coordinates are
`tau + 1j*beta`, with a fixed lens at the origin and source position
`theta_E * ((t-t0)/t_e + 1j*u0)`. The luminous-lens blend is at `0 mas`; the
unrelated blend is fixed at `3 + 2j mas`. Both have `F_b/F_s=0.5` and are assumed
unresolved and unmicrolensed. These positions are illustrative, rather than a
choice of instrumental resolution.

The relatively large source illustrates the transit clearly. All time-series
panels span -40 to +40 days, with dotted lines at limb contact (about +/-9.80
days). Astrometric plots use microarcseconds; saved numeric data retain mas.
Separate track sampling extends to `abs(t-t0)/t_e=1000` to show the wings
approaching the origin without artificially closing the curves.

The shared-axis time-series layout follows
[Lee et al. (2010), Figs. 5–6 and 8](https://arxiv.org/pdf/1005.3021), and the
expanded vertical scale of the source tracks follows
[Mao & Witt (1998), Fig. 2](https://arxiv.org/pdf/astro-ph/9804045).
These figures are new `singlens` calculations for a point lens; Lee et al.
also include finite-lens occultation, which this example does not model.
All figures have a landscape or square overall layout, including the
three-panel time series.

The example-local `blended_observables` function combines the lensed source
centroid and blend light by their observed fluxes. It returns the total flux,
observed centroid, same-time unlensed combined-light centroid, and their
displacement. Thus each blended astrometric curve is relative to its own
instantaneous combined baseline, rather than a fixed reference position or
the source alone. All positions must use the same frame and angular units.
The script checks the flux-weighted identity, the no-blend and unlensed limits,
coincident source/blend positions, identical blended photometry, and JAX
compilation/differentiation of this composition.

See the [astrometry chapter](../../docs/astrometry.md) for equations,
coordinate conventions, and the distinction between a lensing shift and an
observed sky position. This recipe is an observation-model example; it does
not add a public blending API to `singlens`.
