# Benchmark measurement details

[Benchmark results](benchmarks.md) give the accuracy, runtime, and inference
comparisons. This page provides the reference calculation and backend setup.
Run the scripts below from the repository root with `.[examples,validation]` installed:

```bash
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/accuracy.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/benchmark.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/inference_speed.py
```

## Accuracy reference

Integrate in lens-centered polar coordinates. The area element cancels the
point-source magnification's $1/u$ singularity. Each ray intersects the source
on a finite interval; the uniform part has an analytic radial integral and
the limb-darkened part is integrated numerically. This avoids FFTs and
microlensing lookup tables in the reference:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import singlens as sl
```

```{literalinclude} ../example/compare-single-vbml/code/accuracy.py
:language: python
:pyobject: integral
```

The shared separation grid includes source contact and approximation boundaries:

```{literalinclude} ../example/compare-single-vbml/code/accuracy.py
:language: python
:start-at: RATIOS =
:end-at: RADII =
```

For `rho=.01` and `a1=.5`, compare all FFT resolutions against the tighter integral:

```python
rho, a1 = 0.01, 0.5
u = jnp.asarray(RATIOS * rho)
reference = np.array([integral(float(x), rho, a1=a1, tolerance=1e-11) for x in u])
bins = [512, 1024, 2048, 4096, 8192, 16384]
errors = []
for n_fft in bins:
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=n_fft))
    actual = np.asarray(jax.jit(model.magnification_at)(u, rho=rho, a1=a1))
    errors.append(np.max(np.abs(actual / reference - 1)))
```

Repeat for each `rho` in `RADII`. For a uniform source use
`Model(source="uniform")`, omit `a1` in its evaluation, and set `a1=0` in the
reference. Comparing quadrature tolerances `1e-9` and `1e-11` checks the
reference independently of FFT resolution.

## Native batch interfaces

Use the imports above and `median_runtime` from
[the runtime measurement](benchmarks.md#time-the-same-light-curve).
The same event and times are passed to all three native batch interfaces;
only the magnification backend changes. VBM receives natural logarithms,
whereas MulensModel receives physical event parameters and its normalized
limb coefficient `gamma = 2*a1/(3-a1)`.

```python
import VBMicrolensing
import MulensModel as mm

params = sl.Parameters(t0=0.0, u0=0.005, t_e=20.0, rho=0.01, a1=0.5)
model = sl.Model(source="linear")
evaluate = jax.jit(model.magnification)

vbm = VBMicrolensing.VBMicrolensing()
vbm.Tol = vbm.RelTol = 1e-3
vbm.a1, vbm.minannuli = 0.5, 2
# VBM's batch interface takes natural logs in this order.
vbm_params = [np.log(0.005), np.log(20.0), 0.0, np.log(0.01)]

mulens = mm.Model(dict(t_0=0.0, u_0=0.005, t_E=20.0, rho=0.01))
mulens.default_magnification_method = "finite_source_LD_Yoo04"
gamma = 2 * 0.5 / (3 - 0.5)

counts = [10, 30, 100, 300, 1000]
timings = {name: [] for name in ("singlens", "VBM", "MulensModel")}
for count in counts:
    times = np.linspace(-1.0, 1.0, count)
    device_times = jax.block_until_ready(jnp.asarray(times))
    vbm_times = times.tolist()
    calls = {
        "singlens": lambda: evaluate(device_times, params).block_until_ready(),
        "VBM": lambda: vbm.ESPLLightCurve(vbm_params, vbm_times)[0],
        "MulensModel": lambda: mulens.get_magnification(times, gamma=gamma),
    }
    for name, call in calls.items():
        timings[name].append(median_runtime(call))

fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
for name, seconds in timings.items():
    ax.loglog(counts, 1e6 * np.asarray(seconds), "o-", label=name)
ax.set(xlabel="Finite-source evaluations per batch", ylabel="Total batch time [µs]")
ax.legend()
```

The uniform comparison uses `source="uniform"`, VBM `a1=0`, and
MulensModel's `finite_source_uniform_Gould94` method. The recorded experiment
also shuffles measurement order; timing variation between machines is expected.

### VBM's far-field approximation

In VBM 5.6.1, `ESPLMag2` estimates the leading uniform-source correction `Q`.
When `10*Q < Tol`, it returns the point-source magnification plus `Q`;
otherwise it evaluates the finite source. This decision uses the absolute
tolerance `Tol`, rather than a fixed `u/rho` or `RelTol`.
See [the implementation](https://github.com/valboz/VBMicrolensing/blob/main/VBMicrolensing/lib/VBMicrolensingLibrary.cpp#L539-L565).
Its limb-darkening integration also uses `RelTol`. Comparisons retain each
package's normal approximations and assess achieved errors independently.
