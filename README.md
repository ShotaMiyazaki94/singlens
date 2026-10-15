# singlens

<img src="logo_transparent.png" alt="singlens logo" width="280">

Differentiable single-lens microlensing in JAX: static and annual-parallax light
curves, finite sources, limb darkening, and gradient-based inference.

```bash
python -m pip install -e '.[examples]'
```

Requires Python 3.12+ and JAX 0.11.2+. Enable float64 before evaluation.

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from singlens import magnification, AnnualParallax

times = jnp.linspace(7980., 8020., 501)
event = dict(t0=8000., u0=0.005, t_e=20., rho=0.01, a1=0.5)
geometry = AnnualParallax(ra_deg=266., dec_deg=-29., t_ref=8000.)
A = magnification(times, **event, parallax=geometry,
                          pi_e_n=0.2, pi_e_e=-0.1)
```

`magnification` takes event parameters and returns an array with the shape of
`times`. Omit `rho` for a point source; supply `rho` for a uniform disk, or
`rho` and physical `a1` for linear limb darkening. Times above are HJD − 2450000
in days. `AnnualParallax` holds fixed sky geometry; parallax amplitudes remain
fitted inputs.

```python
curve = jax.jit(lambda rho: magnification(times, t0=8000., u0=0.005,
                                        t_e=20., rho=rho, parallax=geometry,
                                        pi_e_n=.2, pi_e_e=-.1))
A = curve(0.01)
dA_drho = jax.jacfwd(curve)(0.01)
```

Use `Model`, `Parameters`, and `fit_flux` for named event parameters,
source-position input, and source/blend flux fitting.
See the [API](docs/api.md) and [examples](example/README.md).
[Benchmarks](docs/benchmarks.md) compare magnification accuracy, CPU runtime,
and inference efficiency with VBMicrolensing and MulensModel.

Build the local documentation with:

```bash
python -m pip install -e '.[docs]'
python -m sphinx -E -b html -W --keep-going docs docs/_build/html
```

Open `docs/_build/html/index.html` in a browser.

## Attribution

Adapted from [microJAX](https://github.com/ShotaMiyazaki94/microjax) and the
FFT method of [Sugiyama (2022)](https://arxiv.org/abs/2203.06637).
See [attribution](docs/attribution.md), `NOTICE`, and `LICENSES/` for citations
and the MIT, Apache-2.0, and BSD-3-Clause license notices.
