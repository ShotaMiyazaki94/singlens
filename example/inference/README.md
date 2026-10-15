# Roman-like parallax injection–recovery

```bash
python -m pip install -e '.[examples]'
python example/inference/code/inference.py --hmc
```

Run from the repository root. The script simulates one 72-day season with
12-minute cadence (8640 observations), then jointly fits a linearly limb-darkened
parallax event and a source-variability GP using tinygp and NumPyro NUTS.
`--warmup` and `--samples` default to 500 and 800 for each of two chains.
Without `--hmc`, only the data and MAP fit are produced.

The [reference experiment](https://github.com/ShotaMiyazaki94/test_microjax/tree/979fa9b33b69c4ba89eb7ad70ebae3d23da349cc/Roman_recovery)
supplies the cadence, W149 source magnitude 24.744, source fraction 0.395,
and magnitude-error table. We assume an **active low-mass dwarf**, with
**1% intrinsic flux RMS, a 5-day rotation period, and 20-day coherence**.
Near-infrared spot modulation on these scales is motivated by
[Goulding et al. (2012)](https://doi.org/10.1111/j.1365-2966.2012.21932.x).
Their periodic M-dwarf variables span 0.16–90 days and 9–115 mmag
peak-to-peak in J. A 1% RMS sinusoid has about 31 mmag peak-to-peak amplitude.
Activity and coherence are explicit scenario choices, not determined by the
apparent magnitude alone. The J-band observations motivate a plausible scale;
this is not a calibrated W149 variability prediction or a population forecast.

The damped-oscillator GP models intrinsic source flux: `F(t) = A(t) * (f_s + g(t)) + f_b`.
The stellar variability is magnified along with the source. At baseline,
blending dilutes its 1% RMS to about 4.3 mmag; at high magnification it approaches
10.9 mmag. This disk-integrated approximation does not model individual spot
crossings. The magnitude-error table sets the white photometric noise.

The event has `t0=.2`, `u0=.005`, `t_e=16.235`, `rho=.01`,
`pi_e_n=.2`, `pi_e_e=-.1`, and `a1=.5`. Times are HJD − 2458000; fixed geometry is
RA 266°, Dec −29°, reference epoch 0. Injection and recovery use 4096 FFT bins.
All twelve event, parallax, limb-darkening, flux, and GP parameters are inferred. The selected
`u0>0` branch does not explore all parallax degeneracies. The fit includes both fluxes and known white-noise variances.

The [inference tutorial](../../docs/tutorials/inference.ipynb) writes out the
GP kernel, NumPyro model and priors, NUTS sampling, conditional GP predictions,
and posterior plots. `gp_sigma` is in
unmagnified source-flux units; `gp_scale` and `gp_period` are the coherence
time and rotation period in days.

- [Recovery](outputs/recovery.png): magnitudes and residuals relative to the
  **true lens curve only**. Both panels show the lens + conditional GP
  posterior median and pointwise 90% interval, excluding new measurement
  noise. Residuals retain the simulated GP and white noise. Dark points show
  0.1-day weighted bins with white-noise-only error bars; all original data
  enter the fit.
- [Corner](outputs/corner.png): all twelve parameters, truth, 68%/95% contours,
  and marginal 5/50/95% quantiles.
- [Trace](outputs/trace.png): both post-warmup chains and truth.

Figures also have PDF versions. `synthetic_event.csv`, `posterior.npz`,
`recovery.csv`, `hmc.json`, and `optimization.json` retain the data, chains,
credible intervals, diagnostics, and initialization. Use `--output-dir PATH`
to change the destination. A single realization does not establish coverage;
independent magnification accuracy is assessed in the [benchmarks](../../docs/benchmarks.md).
