# Accuracy and speed

Install `.[examples,validation]` and run from the repository root:

```bash
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/accuracy.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/benchmark.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/inference_speed.py
```

- `accuracy.py`: magnification error versus FFT bins, checked against
  independent source integration, VBMicrolensing, and MulensModel.
- `benchmark.py`: total magnification runtime versus the number of finite-source
  evaluations in one batch (10–1000, on a logarithmic axis), plus runtime versus
  FFT bins at 100 evaluations, at a common source geometry.
- `inference_speed.py`: **singlens/NUTS versus VBM/emcee** on freshly generated
  Roman-like independent-noise data, with **singlens/emcee** also shown to
  measure the sampler's contribution at a fixed lensing backend.
  All nine event, parallax, limb-darkening,
  and flux parameters are jointly inferred. Each run targets at least 5000
  minimum parameter ESS and must pass convergence checks.

[Benchmark results](../../docs/benchmarks.md) describe the input data,
recovery intervals, accuracy, and inference efficiency.
[Measurement details](../../docs/benchmark-methods.md) give the reference
integral, separation grid, and native batch interfaces. Figures show:

- [Data and recovery](outputs/inference_recovery.png): all three recovered curves
  and magnitude residuals relative to truth, over the season and peak.
- [FFT resolution cost](outputs/runtime_vs_bins.png): singlens runtime versus
  FFT bins for 100 evaluations per batch.
- [Efficiency](outputs/inference_speed.png): ESS versus elapsed time and
  ESS/second for each parameter.
- [Corner comparison](outputs/posterior_comparison.png): all three posteriors,
  68%/95% contours, and truth on shared axes.
- [Trace comparison](outputs/trace_comparison.png): limb darkening and
  event timescale, with each sampler's step axis.

JSON/CSV reports, posterior samples, and PNG/PDF figures are saved in
`outputs/`. Use `--help` for settings, and run timing experiments without
competing intensive work. The [stellar GP recovery](../inference/README.md)
is a separate application example.
