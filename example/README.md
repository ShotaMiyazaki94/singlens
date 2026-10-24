# Examples

Run scripts from the repository root after installing
`python -m pip install -e '.[examples,validation]'`.
Before running physical-source examples, fetch their checked atmosphere inputs
with `python tools/fetch_example_data.py --bootstrap`. The CK04 table is
downloaded locally from STScI and is excluded from the repository and packages;
the helper verifies the original inputs and extracted table against recorded hashes.

| Topic | Run | Result |
| --- | --- | --- |
| Parallax light curves | `python example/lightcurve/code/lightcurve.py` | Point, uniform, and linear sources |
| Astrometry and blending | `python example/astrometry/code/astrometry.py` | Light curves, signed shifts, and centroid tracks with luminous-lens or unrelated blends |
| Event derivatives | `python example/jacobian-single/code/autodiff.py` | Parallax Jacobian, including limb darkening |
| Photometry + GP | See [inference](inference/README.md) | Finite-source event and stellar GP recovery |
| Flux marginalization | `python -m example.flux_marginalization.code.compare` | Same-prior joint and marginalized NUTS fits |
| Photometry + Astrometry | `python example/roman_astrometry/code/fit.py` | Luminous-lens event, annual parallax, and angular Einstein radius |
| Method benchmarks | See [accuracy and speed](compare-single-vbml/README.md) | Table/FFTLog/VBM accuracy and batch runtime |
| Inference benchmark | `python -m example.parallax_inference.code.run_suite` | Annual-parallax event; singlens Table/NUTS versus VBM/emcee, with both fluxes sampled |

See each topic's README for output options and plotting commands. Saved
figures, observations, posterior samples, and reports accompany the
[Photometry + GP tutorial](../docs/tutorials/inference.ipynb) and
[joint recovery example](../docs/roman-astrometry.md).
