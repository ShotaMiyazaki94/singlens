# Benchmarks

Compare **FFTLog**, the bundled **Table**, and **VBMicrolensing** for
magnification accuracy and evaluation speed. The inference comparison fits
a stellar-lens event with annual parallax using NUTS and emcee.
Measurements use CPU and float64; [measurement details](benchmark-methods.md)
record the fixed settings, reference calculations, and timing protocol.
Each method runs alone on the same machine. These are whole-machine timings;
CPU usage and parallel execution settings accompany the reports.

`Model(source="linear")` uses Table;
`Model(source="linear", solver="fft")` selects FFTLog. Neither requires
numerical tuning. For astrometric accuracy and joint evaluation, see
[finite-source effects](finite-source-effects.md#validation).

(accuracy-versus-fft-bins)=
## Accuracy

Compare maximum relative magnification error against independent source
integration, for eight radii from `1e-6` to 10. Each radius has 105 sampled
separations over `0 <= u/rho <= 50`, including alignment and dense sampling
near contact. Uniform sources and linear limb darkening (`a1=.5`) are shown
separately. These are sampled maxima, not uniform error bounds.

```{figure} ../example/compare-single-vbml/outputs/accuracy.png
:alt: Independent magnification errors versus source radius for FFTLog, Table, and VBMicrolensing

One fixed configuration per method. Non-finite external-library values are
excluded from maxima and counted in the report. See
[reference-package settings](benchmark-methods.md#reference-package-settings)
for valid domains and missing points.
```

Table's largest sampled relative magnification error is `1.23e-7` across
these radii and profiles.

Download the {download}`accuracy report <../example/compare-single-vbml/outputs/accuracy.json>`
and {download}`per-point errors <../example/compare-single-vbml/outputs/accuracy.csv>`.
The [FFTLog convergence study](benchmark-methods.md#fftlog-resolution)
examines numerical resolution separately.

## Magnification runtime

Evaluate the same event at 10–1000 epochs per batch, with `rho=.01`,
`u0=.005`, `t_e=20` days, and linear coefficient `a1=.5`.
Observation times span two days around peak, covering
`0.5 <= u/rho <= 5.03`.

```{figure} ../example/compare-single-vbml/outputs/benchmark.png
:alt: Total batch runtime versus epoch count for FFTLog, Table, and VBMicrolensing

Median batch time over 100 synchronized calls after three warmups.
Trajectories and API conversions are included; input preparation, model setup,
and compilation are excluded. Methods retain their native approximations;
achieved accuracy differs as shown above.
```

At 1000 epochs with linear limb darkening, Table is **3.87× faster than
FFTLog** and **6.12× faster than VBM**. For uniform sources, VBM is fastest.

(time-the-same-light-curve)=
To time a light curve, construct the model before the timer, warm up each
input shape, and call `jax.block_until_ready` on every measured result.
The {download}`benchmark script <../example/compare-single-vbml/code/benchmark.py>`
uses the same event and timing protocol for all methods.
Download {download}`timings <../example/compare-single-vbml/outputs/benchmark.json>`
and {download}`raw repeats <../example/compare-single-vbml/outputs/benchmark.csv>`.

(runtime-versus-fft-bins)=
The [FFTLog resolution study](benchmark-methods.md#fftlog-resolution)
records the cost of changing the transform grid.

(inference-efficiency)=
## Inference

Compare complete inference runs for the same annual-parallax event:

| Finite-source calculation | Sampler | CPU execution |
| --- | --- | --- |
| singlens Table | NumPyro NUTS (adaptive HMC) | 16 parallel chains |
| VBMicrolensing | emcee ensemble sampler | 16 worker processes |

singlens uses `Model(source="linear", solver="table")`, interpolating its
packaged finite-source bases. VBM uses its ESPL tables and limb-darkening
integration. JAX autodiff supplies the NUTS posterior gradients.
Both fit **nine parameters**, including source and blend fluxes;
fluxes are sampled jointly. The observations, priors, exposure averaging,
and ESS target are shared. This compares the solver and sampler together;
[measurement details](benchmark-methods.md#inference) specify each implementation.

### Data and posterior

A synthetic stellar-lens event has $t_E=67$ days, $\rho=0.00458$,
$u_0/\rho=0.46$, and $(\pi_{E,N},\pi_{E,E})=(0.096,-0.072)$.
A $0.49\,M_\odot$ lens at 5.48 kpc and a source at 8 kpc give
$\theta_E=0.48$ mas; the source radius is $2.2\,\mu$as.
The setup is inspired by [OGLE-2007-BLG-050](https://doi.org/10.1051/0004-6361/200912923).
The observations are synthetic.

An idealized ground network supplies 1,199 observations over 210 days:
daily wings, hourly shoulders, and five-minute peak coverage. The 13-hour
source transit contains 157 observations. The source has apparent $I=18.5$
and supplies 80% of baseline light. Fixed 1–60 second exposures and known
Gaussian errors include photon, sky, and read noise plus an independent
0.3% fractional floor. Weather and site visibility are not simulated.

```{figure} ../example/parallax_inference/outputs/data.png
:alt: Synthetic I-band observations over the season and finite-source peak, with residuals relative to the injected mean

Independent source integration supplies the exposure-averaged injection.
This realization uses noise seed 43; all observations enter both fits.
```

Infer $t_0,u_0,t_E,\rho,\pi_{E,N},\pi_{E,E}$, the linear limb-darkening
coefficient $a_1$, and fluxes $f_s,f_b$. Both runs use the same priors and
Table-based MAP/Hessian initialization. The [model and priors](benchmark-methods.md#inference)
include the NumPyro definition and the joint Gaussian flux prior.

```{figure} ../example/parallax_inference/outputs/corner.png
:alt: singlens Table with NumPyro NUTS and VBMicrolensing with emcee, showing nine-parameter posteriors and injected values

Posteriors for noise seed 43, conditional on $u_0>0$. Plot thinning affects only the display;
ESS uses all saved chain steps. Download the
{download}`full-size PDF <../example/parallax_inference/outputs/corner.pdf>`.
```

The broad parallax posterior and competing signed solutions limit the
lens-mass constraint. The comparison measures sampling within the positive
branch; it does not measure movement between disconnected modes.
Independent integration at sampled truth and MAP epochs checks both
finite-source solvers. The largest flux errors are `2.8e-6 σ` for Table and
`0.40 σ` for VBM, where $\sigma$ is each observation's uncertainty.
Across the three noise draws, posterior means differ by up to 0.33 pooled
posterior standard deviations. The comparison retains each method's native
numerical settings.

### Sampling efficiency

Report the smallest **effective sample size (ESS) per elapsed second** among
the nine sampled parameters. Both samplers target estimated ESS ≥ 5,000,
with additional [convergence checks](benchmark-methods.md#sampling-protocol).
Total time includes model preparation, shared initialization, compilation,
warmup, and production. Data generation, diagnostics, numerical validation,
posterior selection, and plotting are excluded.

```{figure} ../example/parallax_inference/outputs/sampling.png
:alt: Sampling progress and ESS per elapsed second for each parameter, comparing singlens Table with NumPyro NUTS and VBMicrolensing with emcee

Noise seed 43: cumulative minimum ESS (left) and efficiency by parameter
(right). Time includes setup, compilation, warmup, and sampling.
```

Across three noise realizations on an 18-core Apple M5 Pro, Table + NUTS
achieves **6.2–8.8× higher minimum ESS/s** than VBM + emcee (median **7.8×**).
Using measured CPU time summed over all threads and worker processes,
minimum ESS per CPU-hour is **4.6–7.0× higher** (median **6.0×**).
Total times are 28–47 seconds and 147–283 seconds, respectively. All runs pass
the stated convergence checks. Execution layouts were selected using a
separate noise draw; the reports retain measured CPU usage for both methods.

(compute-ess-per-second)=
Download the {download}`three-seed summary <../example/parallax_inference/outputs/suite.json>`,
{download}`representative report <../example/parallax_inference/outputs/comparison.json>`,
and {download}`input data <../example/parallax_inference/outputs/data.csv>`.
[Measurement details](benchmark-methods.md#inference) provide the stopping
rules, reproduction commands, and unthinned posterior chains.

```{toctree}
:hidden:

benchmark-methods
```
