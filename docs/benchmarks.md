# Benchmarks

Three measurements on CPU, float64: **accuracy versus FFT bins**, **magnification
runtime**, and **inference ESS/second**. FFT resolution and observation count
are measured separately.

The code below shows how to measure runtime and sampling efficiency.
[Measurement details](benchmark-methods.md) give the independent accuracy
reference and complete reproduction commands. Install the comparison packages:

```bash
python -m pip install 'singlens[examples,validation]'
```

Enable float64 before evaluating magnifications:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import singlens as sl
```

## Accuracy versus FFT bins

The reference is independent numerical integration over the source disk.
For each FFT size, the figure shows the maximum relative error
`abs(A / reference - 1)` over the same separations, `0 <= u/rho <= 50`.
It includes six source radii and both uniform and linearly limb-darkened
profiles, with extra sampling near the source edge.
The [reference calculation](benchmark-methods.md#accuracy-reference)
includes the integration code and FFT-bin sweep.

```{image} ../example/compare-single-vbml/outputs/accuracy.png
:alt: Maximum sampled relative error versus FFT bins for six source radii
```

The dashed line marks 0.1%; the `rho=1e-6` and `1e-5` curves overlap.

At `rho=.01`, maximum sampled errors are:

| Implementation | Uniform | Linear (`a1=.5`) |
| --- | ---: | ---: |
| singlens, 512 bins | 1.229% | 0.7374% |
| singlens, 1024 bins | 0.2818% | 0.1956% |
| singlens, 2048 bins | 0.1454% | 0.09789% |
| **singlens, 4096 bins (default)** | **0.09349%** | **0.06208%** |
| singlens, 8192 bins | 0.08832% | 0.05848% |
| singlens, 16384 bins | 0.0882% | 0.05841% |
| VBM, tolerance `1e-3` | 0.1847% | 0.1719% |
| VBM, tolerance `1e-6` | 0.1847% | 0.9872% |
| MulensModel, Gould94/Yoo04 | 0.001251% | 0.001113% |

**Use 4096 bins by default.** Increasing the resolution further gives little
improvement in this comparison; see
[why accuracy levels off](benchmark-methods.md#why-accuracy-levels-off).
A wider check over `1e-6 <= rho <= 10`
finds a largest sampled error of **0.101%**, for a uniform source.
These measurements are not an error bound for every event; see
[numerical accuracy](numerics.md) for the tested range and derivative guidance.

VBM and MulensModel are evaluated against the same independent reference.
VBM's requested tolerance is a stopping criterion, not a bound on its table
errors: tightening it need not improve the maximum error. The 0.9872% entry
occurs very close to alignment. Undefined results at exact alignment are
excluded from the reference packages' maxima. The
[measurement details](benchmark-methods.md#reference-package-settings)
give the settings and unsupported cases.

Download {download}`accuracy report <../example/compare-single-vbml/outputs/accuracy.json>`
and {download}`per-point errors <../example/compare-single-vbml/outputs/accuracy.csv>`.

## Magnification runtime

The horizontal axis is **N, the number of finite-source magnifications
requested in one batch**, on a logarithmic scale: 10, 30, 100, 300, and 1000.
Each batch shares one source radius
and limb coefficient; only the number of queried positions changes. All
methods receive the same positions, sampling a common trajectory over
`0.5 <= u/rho <= 5.03`, with `rho=.01` and `a1=.5` for linear limb darkening.
N counts requested magnifications, including positions where a package uses
an approximation.

singlens uses **4096 FFT bins**; VBM uses **`Tol=RelTol=1e-3`**, with
`minannuli=2` for linear limb darkening. MulensModel uses Gould94/Yoo04.

```{image} ../example/compare-single-vbml/outputs/benchmark.png
:alt: Total time versus number of finite-source magnification evaluations, for uniform and linear sources
```

The vertical axis is **total batch time in microseconds**, on a logarithmic
scale. Lines show median durations after warmup. For linear limb darkening, singlens is faster
than VBM at N=1000 in this run; MulensModel remains faster. Both reference
packages are faster for uniform sources. The different slopes follow from
how the methods share work:

- **singlens:** one FFT magnification grid serves all N positions. At fixed
  FFT resolution, constructing that grid costs the same for every N; querying
  it adds work for each position. Grid construction dominates for small
  batches, so the curve is nearly flat. At larger N, interpolation and array handling also
  contribute. Calling each position separately would repeat the shared work.
- **VBM:** each position requires its own magnification calculation, so total
  time grows roughly with N. Uniform sources use the ESPL table; limb
  darkening adds [radial integration](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/LimbDarkening.md)
  at each position. Its cost also depends on separation and the requested tolerance.
- **MulensModel:** the selected Gould94/Yoo04 methods evaluate tabulated
  corrections for each position in an array. This work grows with N, while
  fixed API overhead makes the increase less pronounced for small batches.
  See the [method documentation](https://rpoleski.github.io/MulensModel/MulensModel.pointlens.html).

The shared FFT grid belongs to **one likelihood evaluation**. In the inference
comparison, `rho` and `a1` are sampled, so each new parameter evaluation
reconstructs the corresponding grid. These timings include that work;
they do not reuse magnification grids between sampler steps.
An event evaluation with **all positions at `u/rho >= 20` skips the FFT**
and uses a far-field expansion. The batches measured here contain closer
positions and build the grid.

Actual errors are evaluated separately above; nominal tolerances do not
guarantee equal achieved accuracy. Each package retains its normal
approximations away from the source.

The native light-curve APIs are used to evaluate each batch, including their
trajectory calculation. Each timing is a median of 100 synchronized repeats after three warmups.
Inputs and models are prepared before timing; intrinsic API conversions remain
included. First-call initialization and compilation are recorded separately.

### Time the same light curve

The important distinction in JAX is between dispatching work and waiting for
its result. Synchronize every measured evaluation, and warm up each input
shape before collecting durations:

```python
from time import perf_counter


def median_runtime(call, repeats=100):
    for _ in range(3):
        call()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        durations.append(perf_counter() - start)
    return np.median(durations)
```

For singlens, compile the light curve and synchronize the returned array.
The [measurement details](benchmark-methods.md#native-batch-interfaces) include the
VBM and MulensModel calls and their parameter conventions.

```python
params = sl.Parameters(t0=0.0, u0=0.005, t_e=20.0, rho=0.01, a1=0.5)
model = sl.Model(source="linear")
evaluate = jax.jit(model.magnification)
counts = [10, 30, 100, 300, 1000]
seconds = []
for count in counts:
    device_times = jax.block_until_ready(jnp.linspace(-1.0, 1.0, count))
    seconds.append(median_runtime(lambda: evaluate(device_times, params).block_until_ready()))
```

Download {download}`timings <../example/compare-single-vbml/outputs/benchmark.json>`
and {download}`raw repeats <../example/compare-single-vbml/outputs/benchmark.csv>`.

### Runtime versus FFT bins

This comparison fixes **N=100 evaluations per batch** and varies only the
FFT resolution, from 512 to 16384 bins. Source geometry and timing procedure
are the same as above. Both axes are logarithmic.

```{image} ../example/compare-single-vbml/outputs/runtime_vs_bins.png
:alt: singlens batch runtime versus FFT bins for uniform and linear limb-darkened sources
```

More FFT bins increase the cost of constructing and interpolating the
magnification grid, even when N stays fixed. Read this plot together with
the accuracy plot above to compare computational cost and numerical error.
The default is **4096 bins**.

Use the same timing function with a fixed 100-point input to measure the
resolution cost separately from batch size:

```python
device_times = jax.block_until_ready(jnp.linspace(-1.0, 1.0, 100))
seconds_by_bins = []
for n_fft in (512, 1024, 2048, 4096, 8192, 16384):
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=n_fft))
    evaluate = jax.jit(model.magnification)
    seconds_by_bins.append(
        median_runtime(lambda: evaluate(device_times, params).block_until_ready())
    )
```

Download the {download}`figure PDF <../example/compare-single-vbml/outputs/runtime_vs_bins.pdf>`,
{download}`timing report <../example/compare-single-vbml/outputs/runtime_vs_bins.json>`,
and {download}`raw repeats <../example/compare-single-vbml/outputs/runtime_vs_bins.csv>`.

## Inference efficiency

Compare **singlens + NUTS (HMC)**, **singlens + emcee**, and
**VBMicrolensing + emcee** on the same Roman-like event, with independent
Gaussian measurement noise and common priors. The two singlens runs measure
the sampler's contribution with the magnification backend held fixed.
VBM/emcee uses the installed VBM numerical library through an
[array adapter](benchmark-methods.md#vbm-inference-interface).
Nine event, parallax, limb-darkening, and flux parameters are jointly inferred.

### Data and recovery

One **72-day season at 12-minute cadence** contains **8640 observations**.
The source has **W149 = 24.744** and contributes **39.5% of the unmagnified
baseline flux**. The blended baseline is magnitude 23.74, and the peak
is 19.01. The reference error curve gives about **50 mmag
at the season edges and 3.4 mmag at peak**; these set the known
Gaussian flux errors. No GP or intrinsic variability enters this data set or likelihood.

The [reference Roman experiment](https://github.com/ShotaMiyazaki94/test_microjax/tree/979fa9b33b69c4ba89eb7ad70ebae3d23da349cc/Roman_recovery)
supplies the cadence and photometric settings. The injected event has
`t0=.2`, `u0=.005`, `t_e=16.235`, `rho=.01`, `a1=.5`, `pi_e_n=.2`, and
`pi_e_e=-.1`; singlens with **4096 FFT bins** generates the true curve.
Times are HJD − 2458000, with a fixed geocentric reference epoch 0,
RA 266°, Dec −29°. The Earth annual-parallax trajectory makes this an
idealized Roman-like example, rather than a complete Roman forecast.
Data seed is 43; the positive-`u0` branch is selected.

Independent Gaussian priors are defined in these coordinates; logarithms are natural:

| Coordinate | Mean | Standard deviation |
| --- | --- | --- |
| `t0` | 0 | .5 day |
| `log(u0)` | `log(.01)` | 1 |
| `log(t_e)` | `log(15)` | .6 |
| `log(rho)` | `log(.01)` | .7 |
| `log(f_s)` | 0 | 1 |
| `f_b` | 0 | 5 |
| `pi_e_n`, `pi_e_e` | 0 | .5 each |
| `logit(a1)` | 0 | 1 |

For either magnification backend, the data term is the same independent
Gaussian log likelihood:

```python
def white_log_likelihood(A, f_s, f_b, flux, flux_error):
    prediction = f_s * A + f_b
    variance = flux_error**2
    return -0.5 * jnp.sum((flux - prediction) ** 2 / variance + jnp.log(2 * jnp.pi * variance))
```

Add the listed log priors to this quantity. `A` comes from singlens or VBM
at the same event parameters and annual-parallax separations.

```{image} ../example/compare-single-vbml/outputs/inference_recovery.png
:alt: Independent-noise Roman-like data and three recovered lens curves over the season and peak
```

Both rows use magnitudes. Residuals subtract **the true lens curve**.
Blue, burgundy, and green curves and bands show each method's posterior median and
pointwise 90% interval. Dark residual points are 0.1-day bins with white-noise
error bars; all original observations enter all three fits.

The source radius, limb darkening, and event timescale are constrained in this
realization. The north parallax component is weakly constrained: its posterior
remains broad, close to its prior.

```{image} ../example/compare-single-vbml/outputs/posterior_comparison.png
:alt: Overlaid nine-parameter corner plot of singlens NUTS, VBM emcee, and singlens emcee samples
```

The corner overlays **singlens/NUTS (blue, solid)**, **VBM/emcee
(burgundy, dashed)**, and **singlens/emcee (green, dotted)**.
Contours enclose 68% and 95%; gray lines mark truth.
Download the {download}`full-size corner PDF <../example/compare-single-vbml/outputs/posterior_comparison.pdf>`.

### Sampling efficiency

Sampler draws are correlated. The **effective sample size (ESS)** estimates
how many independent draws would give the same precision for a parameter
mean. We compare **the smallest ESS among the nine parameters per second**,
so a rapidly sampled parameter cannot hide a slowly sampled one.

Timing includes the initial fit, backend setup (including the VBM adapter build),
JAX compilation, warmup, and sampling.
Every method is charged the same initial-fit cost. Data preparation, plots,
and convergence checks are outside the timer. All runs reach minimum
ESS ≥ 5000 and pass the [stopping checks](benchmark-methods.md#sampling-protocol).
All three methods were measured sequentially on the CPU of a MacBook Pro with
an Apple M5 Pro chip.

```{image} ../example/compare-single-vbml/outputs/inference_speed.png
:alt: Minimum ESS against elapsed time and ESS per second for the nine inferred parameters
```

Left: minimum ESS versus elapsed time, with the target shown by the dashed
line. Large markers mark completed runs; early ESS estimates are less stable.
Right: ESS per second for each parameter.

| Method | Total [s] | Minimum ESS | ESS/s |
| --- | ---: | ---: | ---: |
| singlens + NUTS | 63.6 | 5163 | 81.13 |
| VBM + emcee | 225.6 | 5143 | 22.80 |
| singlens + emcee | 207.8 | 5048 | 24.29 |

singlens/NUTS reaches **3.56× the ESS/s of VBM/emcee** on this event.
This comparison changes both the magnification calculation and the sampler.
Keeping singlens fixed, **NUTS reaches 3.34× the ESS/s of emcee**
for this event. Rates will depend on the data, posterior, and hardware.

Inference time depends on both the cost of each posterior evaluation and the
number of evaluations needed to reach the target ESS. Magnification runtime
alone therefore does not predict sampling efficiency.

These rates compare the stated numerical settings, not matched achieved
accuracy. The [numerical checks](benchmark-methods.md#sampling-protocol)
report the effect of tightening VBM's tolerance and the posterior differences
between backends.

### Compute ESS per second

Keep successive steps in their original order when estimating autocorrelation.
Here a stream means one NUTS chain or one emcee walker. For a parameter
with $M$ streams and $N$ retained steps per stream,
$\mathrm{ESS}=MN/\tau$, where $\tau$ is the integrated autocorrelation time.
The same [autocorrelation estimator](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
is used for both samplers:

```python
import emcee


def sampling_efficiency(chains, total_seconds):
    # Shape: retained steps, streams, physical parameters.
    tau = emcee.autocorr.integrated_time(chains, tol=0)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise ValueError("Extend the chains before interpreting this ESS estimate.")
    ess = chains.shape[0] * chains.shape[1] / tau
    return ess, np.min(ess) / total_seconds


names = ["t0", "u0", "t_e", "rho", "f_s", "f_b", "pi_e_n", "pi_e_e", "a1"]
# nuts_draws[name]: (chains, steps)
chains_nuts = np.stack([nuts_draws[name].T for name in names], axis=-1)
# ensemble_draws: (independent ensembles, steps, walkers, physical parameters)
steps = ensemble_draws.shape[1]
chains_vbm = ensemble_draws.transpose(1, 0, 2, 3).reshape(steps, -1, len(names))

ess_nuts, rate_nuts = sampling_efficiency(chains_nuts, nuts_total_seconds)
ess_vbm, rate_vbm = sampling_efficiency(chains_vbm, vbm_total_seconds)
```

`nuts_draws` and `ensemble_draws` contain every retained step after warmup;
the two time variables contain the total measured durations.
Reshape ensembles and walkers onto the **stream** axis, never the time axis.
The singlens/emcee run uses the same calculation as VBM/emcee.
Emcee walkers in one ensemble are correlated with one another, so convergence
is also checked using separate ensembles; see the
[sampling protocol](benchmark-methods.md#sampling-protocol).
The downloadable emcee samples keep every tenth step for posterior plots.
Use the unthinned chains when reproducing the reported ESS.

```{image} ../example/compare-single-vbml/outputs/trace_comparison.png
:alt: singlens NUTS and VBM emcee traces for limb darkening and event timescale
```

singlens/NUTS chain 0 and VBM/emcee ensemble 0, walker 0, without thinning.
Bottom and top axes count NUTS and emcee production steps; matching horizontal
positions do not represent equal elapsed time. All chains and walkers enter ESS.
Download the {download}`trace comparison PDF <../example/compare-single-vbml/outputs/trace_comparison.pdf>`.

Download the {download}`inference report <../example/compare-single-vbml/outputs/inference_speed.json>`,
{download}`summary CSV <../example/compare-single-vbml/outputs/inference_speed.csv>`,
{download}`input data <../example/compare-single-vbml/outputs/inference_event.csv>`,
{download}`data and prior settings <../example/compare-single-vbml/outputs/inference_data.json>`,
{download}`NUTS draws <../example/compare-single-vbml/outputs/posterior_singlens_nuts.npz>`,
{download}`VBM/emcee draws <../example/compare-single-vbml/outputs/posterior_vbm_emcee.npz>`,
and {download}`singlens/emcee draws <../example/compare-single-vbml/outputs/posterior_singlens_emcee.npz>`.

```{toctree}
:hidden:

benchmark-methods
```
