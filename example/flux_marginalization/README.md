# Flux marginalization

Compare joint and analytically marginalized NUTS fits to the same microlensing
event, using Table magnifications and identical Gaussian flux priors.

From the repository root, with `.[examples,validation]` installed:

```bash
python tools/fetch_example_data.py --bootstrap
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m example.flux_marginalization.code.compare
```

Run without competing intensive work. Results in `outputs/` include timing
and convergence diagnostics, posterior samples, and PNG/PDF light-curve,
efficiency, and corner plots. The corner plot includes source and blend fluxes
reconstructed from their conditional Gaussian posterior. Add `--plot-only` to regenerate
the figures from saved results.
See the [worked example](../../docs/examples/flux-marginalization.md) for the
likelihood, NumPyro model, flux reconstruction, and measured sampling efficiency.
