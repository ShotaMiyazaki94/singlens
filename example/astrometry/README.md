# Photometry, astrometry, and blending

Run from the repository root with the `examples` extra installed:

```bash
python example/astrometry/code/astrometry.py
```

The script writes three synchronized light-curve/centroid movies, numerical
CSV data, and physical configurations to `outputs/`. Use `--output-dir PATH`
to change the destination.
The public repository retains the movies and configurations. The deterministic
CSV grids are generated locally and excluded from Git and source distributions;
movie rendering uses the calculated arrays directly.

| Movie | Example |
| --- | --- |
| `source_animation.gif` | Planetary-mass lens crossing a giant source; point, uniform, and limb-darkened profiles |
| `long_event_animation.gif` | OGLE-2011-BLG-0462; annual parallax and rectilinear motion |
| `blending_animation.gif` | G-dwarf lens and bulge dwarf source; source-only and luminous-lens centroids |

The first movie expands the vertical angular scale by five; the other two
use equal angular scales. Each movie advances uniformly in physical time.
The black-hole example evaluates the published geocentric solution from
[Mróz et al. (2022), Table 2, Sahu+ column](https://arxiv.org/html/2207.10729v2#S5.T2).
The G-dwarf system is synthetic, with a solar-mass lens at 6 kpc and a
solar-like source at 8 kpc.

Shifts are measured from the same-epoch unlensed light centroid: the source
center for source-only predictions, or the combined source-plus-lens
centroid when the lens is luminous. Coordinates use the fixed
`tau + 1j*beta` frame; plots show microarcseconds and CSV files use mas.

See [astrometry](../../docs/astrometry.md) for equations, sky-coordinate
conventions, and the physical assumptions of each example. The
[joint recovery example](../roman_astrometry/README.md) fits noisy
photometry and sky positions for the luminous-lens system.
