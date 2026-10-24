# Photometry, astrometry, and blending

```bash
python example/astrometry/code/astrometry.py
```

Run from the repository root with the `examples` extra installed. The script
enables JAX 64-bit mode and writes three GIFs, numerical CSV data, and physical
configurations to `outputs/`. Override the destination with `--output-dir PATH`.

Every movie puts the light curve on the left and the centroid-shift geometry
on the right. A time cursor and moving markers identify the same epoch in
both panels. Faint curves show complete paths; brighter trails show the
elapsed interval. Each movie has 241 uniformly spaced epochs, played at
20 fps (12.05 seconds), on a 1155 × 440 landscape canvas.

| Movie | Physical example | Time interval |
| --- | --- | --- |
| `source_animation.gif` | Planetary-mass lens crossing a giant source | ±3.2 days |
| `long_event_animation.gif` | Known black hole OGLE-2011-BLG-0462 | ±719.1 days |
| `blending_animation.gif` | G-dwarf lens and bulge dwarf source, on Roman scales | ±170.21 days |

The first example has `u0=0.1`, `t_e=0.8 day`, `rho=0.5`, and
`theta_E=0.012 mas`; the linear profile uses `a1=0.6`. Assuming `DS=8 kpc`
and `pi_rel=0.1 mas` gives `DL=4.44 kpc`, `M=0.185 M_Jup`, and a giant
source of radius `10.3 R_sun`. Dotted lines mark limb contact at ±0.392 days.
The vertical angular scale is expanded by five, following
[Mao & Witt (1998), Fig. 2](https://arxiv.org/pdf/astro-ph/9804045).
Faint tracks extend to `abs(t-t0)/t_e=1000` as ideal rectilinear
continuations, not long-term parallax predictions.

The BH example adopts the complete joint geocentric solution in
[Mróz et al. (2022), Table 2, Sahu+ column](https://arxiv.org/html/2207.10729v2#S5.T2):
`t0=t_ref=5763.3286`, `t_e=239.7 day`, `u0=0.00263`, `theta_E=5.68 mas`,
and `(pi_e_n, pi_e_e)=(0.073, -0.050)`. Times use HJD − 2450000.
The adopted `DS=5.9 kpc` gives `M=7.88 M_sun`, `DL=1.49 kpc`.
The point-source light curve uses a logarithmic magnification axis;
it excludes the neighboring light present in the ground-based photometry.
The movie compares annual parallax with rectilinear motion, using equal
angular scales. It is a calculation from published parameters, not a refit.

The luminous example is an independent stellar system: a `1 M_sun`
G-dwarf lens at `6 kpc` and a solar-like dwarf source at `8 kpc`, with
`R_source=1 R_sun`, `mu_rel_geo=5 mas/yr`, and `u0=0.3`. These imply
`theta_E=0.5825 mas`, `t_e=42.55 day`, and `rho=0.0009979`.
Annual parallax uses a direction 30 degrees east of north,
`(RA, Dec)=(268.07, -29.23)` degrees, and `t0=t_ref=12030`.
For Roman F146, equal intrinsic source/lens luminosity and equal foreground
extinction imply `F_L/F_s=(8/6)^2=16/9`. This is a plausible illustrative
G-dwarf event, not a survey-population median or observation forecast.
The example uses Earth's orbit; it omits Roman's L2 displacement, seasonal
sampling, cadence, and measurement errors. Each light curve is normalized
by its own baseline flux, and both angular axes use the same scale.

Coordinates are the fixed `tau + 1j*beta` frame. Astrometric plots use
microarcseconds; saved numerical data use mas. The cross is **zero lensing
displacement relative to the unlensed baseline at the same epoch**, not the
lens location or a fixed sky position. For the first two movies that
baseline is the source center. For the luminous comparison each curve uses
its own baseline: source-only or combined source-plus-lens centroid.

The example-local `blended_observables` function combines source images and
lens light by their observed fluxes. It returns total flux, observed
centroid, same-time unlensed combined-light centroid, and their displacement.
Source positions follow the same parallax trajectory used for lensing.
The script checks the flux-weighted identity, physical limits, and JAX
compilation/differentiation. Unrelated blends can use the same composition;
their flux and position trajectory belong in each analysis's observation
model. Finite-lens occultation is outside these point-lens examples.

See the [astrometry chapter](../../docs/astrometry.md) for physical assumptions,
sources, equations, and sky-coordinate conventions. This observation-model
recipe does not add a public blending API to `singlens`.
