# Examples

Run scripts from the repository root after installing
`python -m pip install -e '.[examples,validation]'`.

| Topic | Run | Result |
| --- | --- | --- |
| Parallax light curves | `python example/lightcurve/code/lightcurve.py` | Point, uniform, and linear sources |
| Astrometry and blending | `python example/astrometry/code/astrometry.py` | Light curves, signed shifts, and centroid tracks with luminous-lens or unrelated blends |
| Event derivatives | `python example/jacobian-single/code/autodiff.py` | Parallax Jacobian, including limb darkening |
| Roman-like injection–recovery | `python example/inference/code/inference.py --hmc` | Parallax, limb darkening, and stellar GP recovery; corner and traces |
| Benchmarks | See [accuracy and speed](compare-single-vbml/README.md) | FFT-bin accuracy, VBM/MulensModel timings, inference ESS/second |

Every script accepts `--output-dir`; by default it writes to its topic's
`outputs/`. Figures are PNG/PDF. Timings include device/version metadata.
Recorded benchmarks use CPU.
The inference topic also has `data/` for its attributed photometric error table.
