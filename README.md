# singlens

<img src="https://raw.githubusercontent.com/ShotaMiyazaki94/singlens/main/logo_transparent.png" alt="singlens logo" width="280">

Differentiable single-lens microlensing in JAX: static and annual-parallax light
curves, astrometric centroid shifts, finite sources, limb darkening, and
gradient-based inference.

```bash
python -m pip install singlens
```

Requires Python 3.12+ and JAX 0.11.2+. Install `singlens[examples]` for plotting
and inference dependencies. Enable float64 before evaluation.

See the [documentation](https://singlens.readthedocs.io/en/latest/) for
installation, tutorials, and the API.

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

Use the same event parameters for the unresolved images' astrometric shift:

```python
from singlens import astrometric_shift

shift = astrometric_shift(times, **event, parallax=geometry,
                         pi_e_n=0.2, pi_e_e=-0.1)
shift_mas = 1.2 * shift  # For an angular Einstein radius of 1.2 mas.
```

The complex shift is measured from the unlensed source center in the
`tau + 1j*beta` trajectory frame. It assumes a dark point lens and a circular
source. The [astrometry chapter](https://singlens.readthedocs.io/en/latest/astrometry.html)
shows light curves and centroid tracks, the sky rotation, and an observation
model that adds luminous-lens or unrelated blended light. Source proper
motion and absolute parallax are supplied separately. See
[astrometric accuracy](https://singlens.readthedocs.io/en/latest/numerics.html#astrometric-shifts)
before fitting precise positions.

Use `Model`, `Parameters`, and `fit_flux` for named event parameters,
source-position input, and source/blend flux fitting.
See the [API](https://singlens.readthedocs.io/en/latest/api.html) and
[examples](https://github.com/ShotaMiyazaki94/singlens/tree/main/example).
[Benchmarks](https://singlens.readthedocs.io/en/latest/benchmarks.html) compare
magnification accuracy, CPU runtime,
and inference efficiency with VBMicrolensing and MulensModel.

For source installation and development, see
[CONTRIBUTING](https://github.com/ShotaMiyazaki94/singlens/blob/main/CONTRIBUTING.md).

## Attribution

Adapted from [microJAX](https://github.com/ShotaMiyazaki94/microjax) and the
FFT method of [Sugiyama (2022)](https://arxiv.org/abs/2203.06637).
See [attribution](https://singlens.readthedocs.io/en/latest/attribution.html),
[NOTICE](https://github.com/ShotaMiyazaki94/singlens/blob/main/NOTICE), and
[LICENSES](https://github.com/ShotaMiyazaki94/singlens/tree/main/LICENSES) for
citations and the MIT, Apache-2.0, and BSD-3-Clause license notices.
