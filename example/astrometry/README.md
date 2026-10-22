# Photometry, astrometry, and blending

```bash
python example/astrometry/code/astrometry.py
```

Run from the repository root with the `examples` extra installed. The script
enables JAX 64-bit mode and writes three GIFs, numerical CSV data, and event
configurations to `outputs/`. Override the destination with `--output-dir PATH`.

Every movie puts the light curve on the left and the centroid-shift geometry
on the right. A time cursor and distinct moving markers identify the same
epoch in both panels. Faint curves show the complete paths; brighter trails
show the elapsed interval. Each movie has 241 uniformly spaced epochs,
played at 20 fps (12.05 seconds), on a landscape canvas.

- `source_animation.gif`: point, uniform, and linear source profiles in a
  source-crossing event. It covers -80 to +80 days; the vertical angular
  scale is expanded by five to reveal the transit loop.
- `long_event_animation.gif`: a long event with small finite-source effects,
  comparing annual parallax with rectilinear motion. It covers -540 to +540
  days and uses equal angular scales.
- `blending_animation.gif`: the same long parallax event with and without
  luminous-lens light, on the same time interval and equal angular scales.
  Each light curve is normalized by its own baseline flux.

Strong finite-source effects appear only in the first example: `t0=0 day`,
`u0=0.1`, `t_e=20 day`, `rho=0.5`, and `theta_E=1.2 mas`; the linear profile
uses `a1=0.6`. Dotted lines mark limb contact at approximately +/-9.80 days.
Its faint background tracks extend to `abs(t-t0)/t_e=1000`, approaching the
origin without artificially closing the curves.

The long event uses `t0=8000` in HJD - 2450000, `u0=0.35`, `t_e=180 day`,
`rho=0.005`, and `theta_E=1.2 mas`, with a uniform source. Annual parallax
uses `(pi_e_n, pi_e_e)=(0.12, 0.08)`, `(RA, Dec)=(266, -29)` degrees, and
`t_ref=t0`. The luminous-lens comparison uses `F_L/F_s=0.5`.

Coordinates are `tau + 1j*beta`. Astrometric plots use microarcseconds;
saved numerical data use mas. The source-only tracks show the displacement
relative to the unlensed source. The source example's vertical expansion
follows [Mao & Witt (1998), Fig. 2](https://arxiv.org/pdf/astro-ph/9804045).
These are new `singlens` point-lens calculations; finite-lens occultation
is not included.

The example-local `blended_observables` function combines source images and
lens light by their observed fluxes. It returns the total flux, observed
centroid, same-time unlensed combined-light centroid, and their displacement.
For blending, source positions follow the same parallax trajectory in the
lens-relative frame. All positions use the same frame and angular units.
The script checks the flux-weighted identity, physical limits, and JAX
compilation/differentiation. Unrelated blended light can use the same
composition; its flux and position trajectory belong in the observation
model for each analysis.

See the [astrometry chapter](../../docs/astrometry.md) for equations and
coordinate conventions. This is an observation-model recipe; it does not
add a public blending API to `singlens`.
