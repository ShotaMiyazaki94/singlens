# singlens

<img src="https://raw.githubusercontent.com/ShotaMiyazaki94/singlens/main/logo_transparent.png" alt="singlens logo" width="280">

Differentiable single-lens microlensing in JAX: static and annual-parallax light
curves, astrometric centroid shifts, finite sources, limb darkening, and
gradient-based inference.

```bash
python -m pip install singlens
```

Requires Python 3.12+ and JAX 0.11.2+. Install `singlens[examples]` for plotting
and inference dependencies. Enable float64 before constructing models or JAX arrays.

## Quickstart

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from singlens import AnnualParallax, Model, Parameters

times = jnp.linspace(7980., 8020., 501)
geometry = AnnualParallax(ra_deg=266., dec_deg=-29., t_ref=8000.)
model = Model(source="linear", parallax=geometry)
params = Parameters(t0=8000., u0=0.005, t_e=20., rho=0.01, a1=0.5,
                    pi_e_n=0.2, pi_e_e=-0.1)
A = jax.jit(model.magnification)(times, params)
```

`Model` fixes the source profile and geometry; `Parameters` holds fitted values.
Choose `source="point"`, `"uniform"`, `"linear"`, or `"mu2"` explicitly.
Results preserve the shape of `times`. Parallax times are HJD − 2450000 in days.
See [finite-source effects](docs/finite-source-effects.md) for supported
profiles, solver choices, and accuracy.
Differentiate a parameter with JAX:

```python
curve = jax.jit(lambda rho: model.magnification(times, params.replace(rho=rho)))
A = curve(0.01)
dA_drho = jax.jacfwd(curve)(0.01)
```

Use the same event parameters for the unresolved images' astrometric shift:

```python
A, shift = jax.jit(model.joint)(times, params)
shift_mas = 1.2 * shift  # For an angular Einstein radius of 1.2 mas.
```

The shift is relative to the unlensed source center in the `tau + 1j*beta`
trajectory frame, for a dark point lens and a circular source.

## Documentation

- [Getting started](https://singlens.readthedocs.io/en/latest/): installation, light curves, and autodiff.
- [Finite-source effects](docs/finite-source-effects.md): source profiles, methods, and accuracy.
- [Astrometry](https://singlens.readthedocs.io/en/latest/astrometry.html): sky coordinates, source motion, and blended light.
- [Inference examples](https://singlens.readthedocs.io/en/latest/examples/index.html): photometry with a GP, flux marginalization, and joint photometry/astrometry.
- [API](https://singlens.readthedocs.io/en/latest/api.html) and [benchmarks](https://singlens.readthedocs.io/en/latest/benchmarks.html).

For source installation and development, see
[CONTRIBUTING](https://github.com/ShotaMiyazaki94/singlens/blob/main/CONTRIBUTING.md).

## Attribution

Adapted from [microJAX](https://github.com/ShotaMiyazaki94/microjax).
Table generation uses the disk-mixture method of
[Wang (2026)](https://arxiv.org/abs/2607.12483); the FFT backend follows
[Sugiyama (2022)](https://arxiv.org/abs/2203.06637).
See [attribution](https://singlens.readthedocs.io/en/latest/attribution.html),
[NOTICE](https://github.com/ShotaMiyazaki94/singlens/blob/main/NOTICE), and
[LICENSES](https://github.com/ShotaMiyazaki94/singlens/tree/main/LICENSES) for
citations and the MIT, Apache-2.0, and BSD-3-Clause license notices.
