# Photometry + GP

Infer a finite-source event and stellar variability from synthetic Roman-like
photometry. A 0.5-Earth-mass lens crosses an active K dwarf, giving
$t_E=1.77$ hours and $\rho=0.614$. The [tutorial](../../docs/tutorials/inference.ipynb)
shows the priors, NumPyro model, diagnostics, and saved posterior figures.

## Run

Install the example dependencies and run from the repository root:

```bash
python -m pip install -e '.[examples]'
python tools/fetch_example_data.py --bootstrap
python -m example.inference.code.active_ffp --warmup 600 --samples 1600 --progress
python -m example.inference.code.active_plots
python -m example.inference.code.active_spots
```

Two sequential NUTS chains jointly sample nine event, blend, and GP
parameters. `--output PATH` changes the inference destination;
`--input PATH` selects results for plotting. Documentation builds only load
saved products; they do not run NUTS.

## Model

`F(t) = A(t) * (f_s + g(t)) + f_b + noise`: magnification scales both the
source mean and its GP covariance. The fit uses
`Model(source="linear", solver="fft")`, fixes the limb coefficient at 0.4,
and averages over each exposure. Two damped oscillators model rotation and
its first harmonic; tinygp integrates out the latent GP. See
[`active_ffp.py`](code/active_ffp.py) for the likelihood and NumPyro model,
and [`active_coordinates.py`](code/active_coordinates.py) for the sampling
coordinates and prior Jacobian.

A gapless 70.5-day season has 12.1-minute cadence and known Gaussian errors,
including an assumed 1 mmag floor. The tutorial gives the source and detector
assumptions. This is one synthetic realization, not an instrument forecast
or a lens-mass measurement. The [frozen-spot check](outputs/active_ffp/spatial_spot_check.pdf)
tests the scalar-GP approximation with known local variability.

## Saved results

- [Full season](outputs/active_ffp/recovery.png): light curve, variability,
  and residuals.
- [Event zoom](outputs/active_ffp/event_zoom.png): finite-source fit and a
  separately refitted point-source + GP model.
- [Corner](outputs/active_ffp/corner.png): nine-parameter posterior and injection.

Bands include parameter and conditional-GP uncertainty, without new
measurement noise. All observations enter the fit. The separate-MAP
likelihood difference is a diagnostic, not a calibrated detection significance.

`outputs/active_ffp/` contains observations, posterior samples, conditional
predictions, figures, and provenance. The plot loader checks file hashes
and convergence; `--map-only` produces no posterior figures.

The frozen-spot check retains its report and figures publicly; its intermediate
`spatial_spot_check.npz` arrays are regenerated locally. Earlier `inference.py`
results retain only `outputs/optimization.json` and `outputs/hmc.json` summaries.
Raw observations, the posterior archive, and figures from that earlier run
remain local and are excluded from Git and source distributions.
