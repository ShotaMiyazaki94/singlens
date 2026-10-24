# FFTLog reference results

These JSON summaries preserve the FFTLog-centered comparison preceding the
three-method benchmark. They measure the explicit FFTLog backend, not the
bundled-table model. `manifest.json` records the original artifact names,
sizes, and hashes. Historical CSV data, posterior archives, and rendered
figures remain local and are excluded from Git and source distributions.
Reports contain numerical settings, software versions, seeds, and timing scope.

- `accuracy.json`: FFT-resolution convergence and independent reference errors.
- `benchmark.json` and `runtime_vs_bins.json`: batch timings and resolution cost.
- `inference_*.json`: the FFTLog/NUTS, VBM/emcee, and FFTLog/emcee comparison.

The maintained FFTLog implementation is selected with `solver="fft"`.
Its relation to Sugiyama (2022), including the compensated filter and endpoint
corrections, is described in
[numerical maintenance](../../../../CONTRIBUTING.md#fft-domain-and-regularization).
These are singlens measurements, not a bitwise reproduction of the paper's code.
For current measurements and posterior samples, see
[the benchmark outputs](../../outputs/) and
[measurement details](../../../../docs/benchmark-methods.md).
