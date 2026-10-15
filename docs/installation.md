# Installation

Requires Python 3.12+ and JAX 0.11.2+. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[examples]'
python example/lightcurve/code/lightcurve.py
```

The runtime depends on JAX and NumPy. Optional extras are `examples` (plotting,
optimization, NumPyro, tinygp), `validation` (VBMicrolensing, MulensModel, emcee), and
`docs` (local documentation). Before evaluating finite sources:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

singlens does not change precision or device globally. For accelerators, use
[JAX's installation guide](https://docs.jax.dev/en/latest/installation.html).
The benchmarks use CPU.
