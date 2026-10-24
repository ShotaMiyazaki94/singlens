# Inference examples

Fit stellar variability, integrate over source and blend fluxes, or combine
photometry and astrometry. Each example includes its likelihood, priors,
NumPyro model, and posterior figures.

| Example | Finite-source calculation | Sampler |
| --- | --- | --- |
| [Photometry + GP](../tutorials/inference.ipynb) | singlens FFTLog, linear limb darkening | NumPyro NUTS |
| [Flux marginalization](flux-marginalization.md) | singlens Table, linear limb darkening | NumPyro NUTS |
| [Photometry + Astrometry](../roman-astrometry.md) | singlens FFTLog, uniform source | NumPyro NUTS |

```{toctree}
:hidden:
:maxdepth: 1

../tutorials/inference
flux-marginalization
../roman-astrometry
```
