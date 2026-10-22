# Photometry, astrometry, and blending

```bash
python example/astrometry/code/astrometry.py
```

Run from the repository root with the `examples` extra installed. The script
enables JAX 64-bit mode and writes PNG/PDF figures, CSV data, and the physical
configuration to `outputs/`. Override the destination with `--output-dir PATH`.

- `source_profiles`: compare point, uniform, and linearly limb-darkened sources
  for a source-crossing event. The light curve and signed astrometric components
  show the crossing; the two-dimensional track includes the wider event and a
  central inset. The track's horizontal and vertical axis scales differ.
- `blending`: compare the unblended source, light from the lens, and light from
  an unrelated star. The two blends have the same flux and therefore identical
  photometric light curves, but different astrometric tracks.

The event has `t0=0 day`, `u0=0.03`, `t_e=20 day`, `rho=0.1`, and
`theta_E=1.2 mas`; the linear profile uses `a1=0.6`. Coordinates are
`tau + 1j*beta`, with a fixed lens at the origin and source position
`theta_E * ((t-t0)/t_e + 1j*u0)`. The luminous-lens blend is at `0 mas`; the
unrelated blend is fixed at `3 + 2j mas`. Both have `F_b/F_s=0.5` and are assumed
unresolved and unmicrolensed. These positions are illustrative, rather than a
choice of instrumental resolution.

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
