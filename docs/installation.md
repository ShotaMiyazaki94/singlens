# Installation

Requires Python 3.12+ and JAX 0.11.2+. Install the package in your Python environment:

```bash
python -m pip install singlens
```

JAX and NumPy are installed as dependencies. To add plotting, optimization,
NumPyro, tinygp, and corner for the tutorials and examples:

```bash
python -m pip install 'singlens[examples]'
```

Use a [virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)
if you want to keep these dependencies separate from other projects.

To install the latest source from GitHub:

```bash
python -m pip install 'singlens @ git+https://github.com/ShotaMiyazaki94/singlens.git'
```

Editable installation (`-e`) is only needed when modifying the source; see
[development](https://github.com/ShotaMiyazaki94/singlens/blob/main/CONTRIBUTING.md).

Enable float64 before creating JAX arrays or evaluating magnifications:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import singlens as sl

times = 8000.0 + jnp.linspace(-20.0, 20.0, 101)
geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)
A = sl.magnification(
    times,
    t0=8000.0,
    u0=0.005,
    t_e=20.0,
    rho=0.01,
    a1=0.5,
    parallax=geometry,
    pi_e_n=0.2,
    pi_e_e=-0.1,
)
print(A.shape)  # (101,)
```

Continue with [light curves](tutorials/quickstart.ipynb),
[derivatives](tutorials/autodiff.ipynb), or [NumPyro inference](tutorials/inference.ipynb).

singlens does not change precision or device globally. For accelerators, use
[JAX's installation guide](https://docs.jax.dev/en/latest/installation.html).
The benchmarks use CPU.
