# Method benchmarks

Compare FFTLog, the bundled Table, and VBMicrolensing at one fixed setting
per method. Install `.[examples,validation]` and run sequentially from the
repository root:

```bash
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/accuracy.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/benchmark.py
```

`accuracy.py` measures errors against independent source integration;
`benchmark.py` measures synchronized native batch runtimes. Both save
JSON/CSV reports and PNG/PDF figures in `outputs/`. Use `--output-dir PATH`
for another destination. FFT resolution diagnostics are saved separately as
`accuracy_fft_resolution.*` and `runtime_vs_bins.*`.

The [inference benchmark](../parallax_inference/README.md) fits a stellar-lens
event with annual parallax using singlens Table/NumPyro NUTS and VBM/emcee,
with source and blend fluxes sampled jointly:

```bash
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m example.parallax_inference.code.run_suite
```

See [results](../../docs/benchmarks.md) and
[measurement details](../../docs/benchmark-methods.md) for settings, timing
scope, convergence checks, and downloads. Run timings without competing
intensive work. Earlier FFT-centered JSON summaries and artifact hashes remain
in the [FFTLog reference archive](reference/fftlog/README.md); historical raw
data, posterior archives, and figures are kept local.
