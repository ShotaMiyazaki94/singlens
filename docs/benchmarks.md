# Benchmarks

Three measurements on CPU, float64: **accuracy versus FFT bins**, **magnification
runtime**, and **inference ESS/second**. FFT resolution and observation count
are measured separately.

The code below shows how each quantity is calculated. Install
`.[examples,validation]` to include the reference packages, and enable float64:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import singlens as sl
```

## Accuracy versus FFT bins

The reference is independent numerical source integration. Tightening its
tolerances changes the result by at most **2.42e-12 relative** on the test grid.
Six source radii (`1e-6`, `1e-5`, `1e-4`, `.01`, `.1`, `1`) are evaluated at
the same 105 `u/rho` values from 0 to 50 for every FFT size, with fine sampling
near source contact. These are sampled errors,
not a bound over all separations.

### Independent source integration

Integrate in lens-centered polar coordinates. The area element cancels the
point-source magnification's $1/u$ singularity. Each ray intersects the source
on a finite interval; the uniform part has an analytic radial integral and
the limb-darkened part is integrated numerically. This avoids FFTs and
microlensing lookup tables in the reference:

```python
from scipy.integrate import quad


def reference_magnification(u, rho, a1=0.0, tolerance=1e-11):
    ratio = abs(u) / rho

    def angular(theta):
        transverse = ratio * np.sin(theta)
        half_chord = np.sqrt(max(0.0, 1 - transverse**2))
        center = ratio * np.cos(theta)
        lo, hi = max(0.0, center - half_chord), center + half_chord
        p, q = rho * lo, rho * hi
        hp, hq = np.hypot(p, 2.0), np.hypot(q, 2.0)
        # Stable difference of the uniform radial antiderivative.
        disk = 0.5 * (hi - lo) * (hq + p * (q + p) / (hq + hp))
        if a1 == 0:
            return disk

        def radial(x):
            mu = np.sqrt(max(0.0, 1 - (x - center) ** 2 - transverse**2))
            return mu * ((rho * x) ** 2 + 2) / np.hypot(rho * x, 2.0)

        limb = quad(radial, lo, hi, epsabs=0.01 * tolerance, epsrel=tolerance, limit=150)[0]
        return (1 - a1) * disk + a1 * limb

    upper = np.pi if ratio < 1 else np.arcsin(1 / ratio)
    area_integral = quad(angular, 0.0, upper, epsabs=0.01 * tolerance, epsrel=tolerance, limit=150)[
        0
    ]
    return 2 * area_integral / (np.pi * rho * (1 - a1 / 3))
```

For example, measure maximum relative error for `rho=.01`, `a1=.5`:

```python
ratios = np.unique(
    np.r_[
        0.0,
        1e-6,
        0.00999,
        0.01,
        0.01001,
        0.1,
        0.3,
        0.5,
        0.7,
        0.85,
        0.9,
        0.95,
        0.98,
        0.99,
        0.995,
        1.0,
        1.005,
        1.01,
        1.02,
        1.05,
        1.1,
        1.2,
        1.5,
        2.0,
        3.0,
        5.0,
        10.0,
        19.99,
        20.0,
        20.01,
        50.0,
        np.linspace(0.98, 1.02, 81),
    ]
)
rho, a1 = 0.01, 0.5
u = jnp.asarray(ratios * rho)
reference = np.array([reference_magnification(float(x), rho, a1) for x in u])
bins = [512, 1024, 2048, 4096, 8192, 16384]
errors = []
for n_fft in bins:
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=n_fft))
    actual = np.asarray(jax.jit(model.magnification_at)(u, rho=rho, a1=a1))
    errors.append(np.max(np.abs(actual / reference - 1)))
```

Repeat for the other radii. For a uniform source use `Model(source="uniform")`,
omit `a1` in its evaluation, and set `a1=0` in the reference. Tightening
the quadrature tolerance checks convergence of the reference; increasing
FFT resolution checks the singlens approximation. The full figure collects
these errors for all six radii:

```{image} ../example/compare-single-vbml/outputs/accuracy.png
:alt: Maximum sampled relative error versus FFT bins for six source radii
```

The dashed line marks 0.1%; the `rho=1e-6` and `1e-5` curves overlap.

At `rho=.01`, maximum sampled errors are:

| Implementation | Uniform | Linear (`a1=.5`) |
| --- | ---: | ---: |
| singlens, 512 bins | 1.2290% | 0.7374% |
| singlens, 1024 bins | 0.2818% | 0.1956% |
| singlens, 2048 bins | 0.1454% | 0.0979% |
| **singlens, 4096 bins (default)** | **0.0935%** | **0.0621%** |
| singlens, 8192 bins | 0.0883% | 0.0585% |
| singlens, 16384 bins | 0.0882% | 0.0584% |
| VBM, tolerance `1e-3` | 0.1847% | 0.1719% |
| VBM, tolerance `1e-6` | 0.1847% | 0.9872% |
| MulensModel, Gould94/Yoo04 | 0.00125% | 0.00111% |

**Use 4096 bins by default.** A separate check of 33,666 values at 31 radii
from `1e-6` to 10 finds a largest relative error of **0.101%**, for a uniform source. It includes
near alignment, source contact, approximation boundaries, and far wings.
Uniform and pure linear limb darkening bracket physical `0<=a1<=1`;
the pure quadratic profile is checked separately.
See [numerical accuracy](numerics.md) for limits and derivative guidance.

VBM is checked against the same reference, rather than assumed exact:
[its single-lens implementation uses precomputed ESPL tables](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/SingleLenses.md).
Both `Tol` (absolute) and `RelTol` (relative) are set to the displayed tolerance;
linear limb darkening uses `minannuli=2`. These are stopping criteria, not
achieved error bounds. The uniform-source table error near contact remains
when the tolerance is tightened; the 0.9872% linear-source value above occurs
very close to alignment, at `u/rho=1e-6`.
The `rho=1e-6` rows lie below VBM's documented table range. At exact alignment,
VBM's limb-darkened call is non-finite here and MulensModel's table methods
raise; these points are explicitly marked missing and excluded from their
error maxima (104 valid points versus singlens's 105). Uniform VBM has all 105.
MulensModel's small-source methods are especially accurate at this radius,
but are approximations for larger sources; all radii and RMS errors remain
in the report.

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
an internal approximation. It does not count FFT bins or repeated timing calls.

singlens uses **4096 FFT bins**; VBM uses **`Tol=RelTol=1e-3`**, with
`minannuli=2` for linear limb darkening. MulensModel uses Gould94/Yoo04.

```{image} ../example/compare-single-vbml/outputs/benchmark.png
:alt: Total time versus number of finite-source magnification evaluations, for uniform and linear sources
```

The vertical axis is **total batch time in microseconds**, on a logarithmic
scale. Lines show warm medians. For linear limb darkening, singlens is faster
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
they do not reuse magnification grids between sampler steps. Parameter-independent
transforms are prepared with the model. The inference advantage also depends
on how efficiently NUTS uses gradients to explore the posterior.

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

Here is the linear limb-darkened case for all three native batch interfaces.
Parameter conversions and input allocation happen before the timed calls:

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
for n_fft in bins:
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

Compare **singlens + NUTS (HMC)** and **VBMicrolensing + emcee** on the same
Roman-like event, with independent Gaussian measurement noise and common priors.
Nine event, parallax, limb-darkening, and flux parameters are jointly inferred.
The [stellar GP recovery](tutorials/inference.ipynb) is a separate application.

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
at the same event parameters and annual-parallax separations. The
[NumPyro tutorial](tutorials/inference.ipynb) shows how to define the event
priors and run NUTS; this benchmark uses the white-noise likelihood above
without GP parameters or a latent source process.

```{image} ../example/compare-single-vbml/outputs/inference_recovery.png
:alt: Independent-noise Roman-like data and both recovered lens curves over the season and peak
```

Both rows use magnitudes. Residuals subtract **the true lens curve**.
Blue and orange curves and bands show each method's posterior median and
pointwise 90% interval. Dark residual points are 0.1-day bins with white-noise
error bars; all original observations enter both fits.

| Parameter | Injected | NUTS median | Central 90% interval |
| --- | ---: | ---: | --- |
| `t0` [day] | 0.2 | 0.199966 | [0.199819, 0.20012] |
| `u0` | 0.005 | 0.00495793 | [0.0048861, 0.00503196] |
| `t_e` [day] | 16.235 | 16.3172 | [16.1306, 16.5088] |
| `rho` | 0.01 | 0.00992611 | [0.0098077, 0.0100466] |
| `f_s` | 1.2659 | 1.25855 | [1.24293, 1.2743] |
| `f_b` | 1.93891 | 1.94418 | [1.93249, 1.95535] |
| `pi_e_n` | 0.2 | -0.0484307 | [-0.869323, 0.769955] |
| `pi_e_e` | -0.1 | -0.100841 | [-0.18581, -0.0150048] |
| `a1` | 0.5 | 0.48083 | [0.447303, 0.513636] |

The source radius, limb darkening, and event timescale are constrained in this
realization. The north parallax component remains prior-dominated; including
both components does not imply precise measurement of both. One injection
is not a coverage test.

```{image} ../example/compare-single-vbml/outputs/posterior_comparison.png
:alt: Overlaid nine-parameter corner plot of singlens NUTS and VBM emcee samples
```

The corner overlays **singlens/NUTS (blue, solid)** and **VBM/emcee
(orange, dashed)**. Contours enclose 68% and 95%; gray lines mark truth.
Download the {download}`full-size corner PDF <../example/compare-single-vbml/outputs/posterior_comparison.pdf>`.

### Sampling efficiency

Each method starts from the same joint MAP fit, obtained from the data and
priors without using the injected parameters. Its measured cost is charged
to every method. Timing starts at that fit and includes likelihood
compilation, warmup, and production. Input/model preparation, diagnostics,
IO, plotting, and numerical validation are excluded.

All runs require **minimum parameter ESS ≥ 5000** and convergence checks.
NUTS uses two sequential chains with 500 warmup steps and
retains 1800 draws per chain. VBM/emcee uses two independent ensembles,
32 walkers each, 1000 warmup steps, and
8500 production steps per ensemble. It requires at least
50 estimated correlation lengths and less than 10% change in every
parameter's autocorrelation estimate. Independent ensemble means must agree
within four combined Monte Carlo standard errors; walkers are not treated
as independent chains for R̂. NUTS requires R̂<1.01 and no divergences.

The common [autocorrelation estimator](https://emcee.readthedocs.io/en/stable/tutorials/autocorr/)
uses physical parameter series and `ESS = draws / tau`. The minimum over
all nine parameters sets ESS/second. VBM uses `Tol=RelTol=1e-3` and
`minannuli=2` for linear limb darkening.

### Compute ESS per second

Keep the time axis when estimating autocorrelation. For a physical parameter
with $M$ streams and $N$ retained steps per stream,
$\mathrm{ESS}=MN/\tau$, where $\tau$ is the integrated autocorrelation time.
The following code uses exactly the same estimator for both samplers:

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

`nuts_draws` and `ensemble_draws` are the unthinned production arrays in the
listed physical-parameter order;
`nuts_total_seconds` and `vbm_total_seconds` include the measured common fit,
compilation, warmup, and production. Reshape ensembles and walkers onto the
**stream** axis, never the time axis. This calculation estimates ESS; the
length/stability checks and independent-ensemble comparisons above are still
needed. Emcee walkers within one ensemble are not independent chains for R̂.
Negative autocorrelation can give $\tau<1$, so an ESS larger than the raw
draw count is possible.

The downloadable VBM samples are thinned by 10 for posterior plots. The
reported ESS uses unthinned production chains; use those when reproducing
the calculation above.

```{image} ../example/compare-single-vbml/outputs/inference_speed.png
:alt: Minimum ESS against elapsed time and ESS per second for the nine inferred parameters
```

Left: ESS at measured wall-time checkpoints; early estimates are provisional.
The dashed line marks the common target, and large markers mark completed
runs. Right: whole-run efficiency for each parameter.

| Method | Total [s] | Minimum ESS | ESS/s | Estimated time to ESS 5000 [s] |
| --- | ---: | ---: | ---: | ---: |
| singlens + NUTS | 58.4 | 5065 | 86.76 | 57.8 |
| VBM + emcee | 492.7 | 5289 | 10.73 | 468.7 |

The last column keeps measured MAP and warmup costs and rescales production
time using the final autocorrelation estimate. It estimates time to equal
ESS; the actual validated runs finish at the displayed ESS values.
NUTS has maximum R̂ **0.9997** and **0 divergences**;
VBM's ensemble means agree within **2.37**
combined Monte Carlo standard errors. Both pass their stopping checks.

In this experiment, singlens/NUTS reaches **8.08× the whole-run ESS/s
of VBM/emcee**. A supplementary singlens/emcee control reaches
**24.90 ESS/s**, giving a **3.48× NUTS/control** ratio
on the same numerical backend. Its diagnostics and posterior checks are in
the report. These are results for one event and noise realization, rather
than universal sampler speedups.

The numerical posteriors differ: the largest VBM/NUTS mean shift is
**0.185 pooled posterior standard deviations**, in `rho`
(10.39 combined Monte Carlo standard errors). Across 64 common states,
the VBM-minus-singlens log posterior ranges from -1.167 to
1.269. Tightening VBM to `1e-6`, with `minannuli=4`, changes
its log posterior by -1.834 to
1.874 on the NUTS draws. Magnification
tolerances do not bound accumulated likelihood error. The same-backend
NUTS/emcee means agree within 1.33 combined Monte Carlo standard errors.

At one posterior state, warm evaluation costs **0.61 ms** for
singlens and **0.88 ms** for VBM. The singlens value and gradient
cost **1.22 ms**. These costs and sampler mixing jointly
determine inference efficiency.

```{image} ../example/compare-single-vbml/outputs/trace_comparison.png
:alt: NUTS and VBM emcee traces for limb darkening and event timescale
```

NUTS chain 0 and VBM/emcee ensemble 0, walker 0, without thinning. Bottom
and top axes count each sampler's production steps; matching horizontal
positions do not represent equal elapsed time. All chains and walkers enter ESS.

Download the {download}`inference report <../example/compare-single-vbml/outputs/inference_speed.json>`,
{download}`summary CSV <../example/compare-single-vbml/outputs/inference_speed.csv>`,
{download}`input data <../example/compare-single-vbml/outputs/inference_event.csv>`,
{download}`data and prior settings <../example/compare-single-vbml/outputs/inference_data.json>`,
{download}`NUTS draws <../example/compare-single-vbml/outputs/posterior_singlens_nuts.npz>`,
and {download}`VBM/emcee draws <../example/compare-single-vbml/outputs/posterior_vbm_emcee.npz>`.
