# Photometry + Astrometry

Infer a luminous G-dwarf lens's angular Einstein radius and mass from
synthetic fluxes and east/north centroid positions. The
[joint inference chapter](../../docs/roman-astrometry.md) shows the observation
model, priors, NumPyro model, and posterior figures.

## Run

Install the example dependencies and run from the repository root:

```bash
python -m pip install -e '.[examples]'
python example/roman_astrometry/code/fit.py
python example/roman_astrometry/code/plots.py --overlay
```

`fit.py` optimizes and samples each sign of $u_0$ separately. The positive
branch uses two NUTS chains, 800 warmup steps and 1200 retained draws each.
Documentation builds display saved results without running the sampler.

## Model

The fit uses `Model(source="uniform", solver="fft", parallax=...)` with
$\rho=\theta_*/\theta_E$ and an externally fixed source angular radius.
Source and lens fluxes are inferred independently; no stellar
mass–luminosity relation is imposed. Source size is unconstrained in this
nontransiting event.

The likelihood fits full observed centroid positions. Five baseline
position, proper-motion, and parallax coefficients are integrated out with
proper Gaussian priors; conditional baseline draws enter predictions.
[`fit.py`](code/fit.py) contains the normalized likelihood and NumPyro model.

Six idealized observing windows span 4.5 years. Known Gaussian errors are
independent, with no fitted jitter or variability GP. Earth-orbit parallax,
the assumed errors, and the Roman-inspired schedule define a controlled
experiment, not a calibrated Roman L2 forecast.

## Saved results

`outputs/` contains observations, posterior samples, predictions, diagnostics,
and PNG/PDF figures. The main figures show the preferred positive branch;
`--overlay` adds a conditional comparison of both signs, without assigning
equal branch probabilities. The displayed shift is measured from the fitted
same-epoch **unlensed source-plus-lens centroid**.
