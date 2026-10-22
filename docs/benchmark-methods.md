# Benchmark measurement details

[Benchmark results](benchmarks.md) give the accuracy, runtime, and inference
comparisons. This page provides the reference calculation and backend setup.
Run the scripts below from the repository root with `.[examples,validation]` installed:

```bash
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/accuracy.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/benchmark.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/inference_speed.py
```

The inference benchmark automatically builds a VBM array adapter. It requires
a C++17 compiler and a VBMicrolensing 5.x wheel on macOS or Linux.
See [VBM inference interface](#vbm-inference-interface) for the alternative
Python interface.

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
reference. Comparing quadrature tolerances `1e-9` and `1e-11` changes the
reference by at most `2.42e-12` relative on this grid. Each FFT size uses the
same 105 separations and six source radii (`1e-6`, `1e-5`, `1e-4`, `.01`,
`.1`, `1`).

### Why accuracy levels off

singlens applies a fixed high-frequency damping filter in its FFTLog
calculation. Increasing `n_fft` refines the grid but leaves this filter
unchanged. Once discretization error is small, the remaining smoothing bias
near source contact ($u/\rho\simeq1$) limits further accuracy gains.

### Reference package settings

VBM uses precomputed ESPL tables. Both `Tol` (absolute) and `RelTol`
(relative) are set to the tolerance displayed in the results table;
linear limb darkening uses `minannuli=2`. These control the calculation's
stopping rule and do not bound the table's approximation error. The
uniform-source error near contact remains when the tolerance is tightened.
The `0.9872%` linear-source error at tolerance `1e-6` occurs at
`u/rho=1e-6`, very close to alignment.

The `rho=1e-6` cases are below VBM's documented table range. At exact
alignment, VBM's limb-darkened call is non-finite in this experiment and
MulensModel's table methods raise. Their maxima therefore use 104 valid
points; uniform VBM and singlens use all 105. The downloadable reports mark
missing points explicitly. See VBM's
[single-lens documentation](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/SingleLenses.md).

MulensModel uses Gould94 for uniform sources and Yoo04 for linear limb
darkening. These small-source approximations are especially accurate at
`rho=.01`, but their accuracy changes for larger sources. All tested radii
are included in the [accuracy report](benchmarks.md#accuracy-versus-fft-bins).

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

## VBM inference interface

The default `--vbm-interface array` passes annual-parallax separations to the
installed library's `ESPLMag2` using NumPy arrays. VBM calls are serial, with
the numerical core, ESPL table, and tolerance settings unchanged. Numerical
agreement with the stock Python interface is checked on posterior states.

Use `--vbm-interface python` to select the stock `ESPLLightCurve` interface.
The standalone magnification benchmark always uses native light-curve APIs;
the inference results identify the selected interface in the JSON report.

## Sampling protocol

All three runs start from one fit that maximizes the singlens posterior
(the maximum a posteriori, or MAP, fit), using the data and stated priors.
Injected parameter values are not used for initialization. The fit's
measured cost is included in every method's total time. Timing also includes
backend setup (including one VBM adapter build and load), likelihood compilation,
warmup, and production sampling. Preparing inputs,
convergence checks, file IO, plots, and numerical validation are outside
the timer. VBM uses `Tol=RelTol=1e-3` and `minannuli=2`.

| Method | Independent runs | Warmup per run | Retained steps per run |
| --- | --- | ---: | ---: |
| singlens + NUTS | 2 sequential chains | 500 | 1700 |
| VBM + emcee | 2 ensembles, each with 32 walkers | 1000 | 8000 |
| singlens + emcee | 2 ensembles, each with 32 walkers | 1000 | 8500 |

Every method must reach minimum parameter ESS ≥ 5000. NUTS also requires
R̂ < 1.01 and no divergences. For emcee, production must cover at least 50
estimated autocorrelation times, and each parameter's autocorrelation
estimate must change by less than 10% as sampling is extended.
Separate ensemble means must agree within four combined Monte Carlo
standard errors, which estimate uncertainty due to finite sampling.
Walkers within an ensemble are not treated as independent chains for R̂.

ESS is computed from every retained step in physical parameter coordinates
using the same autocorrelation estimator for all methods. Negative
autocorrelation can make ESS larger than the retained draw count. The emcee files
available for download keep every tenth step for plotting; rerun the script
to obtain the unthinned chains used for ESS. Seeds, package versions,
timings, and diagnostics are recorded in the
{download}`inference report <../example/compare-single-vbml/outputs/inference_speed.json>`.

Numerical accuracy is checked separately from sampler convergence. Tightening
VBM to `Tol=RelTol=1e-6`, `minannuli=4` changes the log posterior by about
−1.7 to +1.7 on sampled states. The largest singlens–VBM posterior mean
difference is 0.17 pooled posterior standard deviations. The speed comparison
therefore uses the stated settings without claiming equal achieved accuracy.
