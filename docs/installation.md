# Installation

Requires Python 3.12+ and JAX 0.11.2+. Install the package in your Python environment:

```bash
python -m pip install singlens
```

For the tutorials and examples, install the optional dependencies:

```bash
python -m pip install 'singlens[examples]'
```

To install the latest source from GitHub:

```bash
python -m pip install 'singlens @ git+https://github.com/ShotaMiyazaki94/singlens.git'
```

For an editable source installation, see
[development](https://github.com/ShotaMiyazaki94/singlens/blob/main/CONTRIBUTING.md).
Before running the source checkout's physical-source examples, tests, or
documentation build, prepare their checked local atmosphere inputs:

```bash
python tools/fetch_example_data.py --bootstrap
```

This downloads the CK04 inputs from STScI and verifies their recorded hashes.
The derived data are excluded from the repository and source distributions;
the lensing library itself does not require these example inputs.

Enable float64 before constructing models or JAX arrays:

```python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import singlens as sl

times = jnp.linspace(-20.0, 20.0, 101)
model = sl.Model(source="linear")
params = sl.Parameters(t0=0., u0=.005, t_e=20., rho=.01, a1=.5)
A = model.magnification(times, params)
print(A.shape)  # (101,)
```

Continue with [light curves](tutorials/quickstart.ipynb),
[derivatives](tutorials/autodiff.ipynb), or
[photometry with a Gaussian process](tutorials/inference.ipynb).

singlens does not change precision or device globally. For accelerators, use
[JAX's installation guide](https://docs.jax.dev/en/latest/installation.html).
