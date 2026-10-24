# Benchmark measurement details

Reproduce the [benchmarks](benchmarks.md) from a source checkout:

```bash
python -m pip install -e '.[examples,validation]'
python tools/fetch_example_data.py --bootstrap
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/accuracy.py
JAX_PLATFORMS=cpu python example/compare-single-vbml/code/benchmark.py
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m example.parallax_inference.code.run_suite
```

Run sequentially without competing intensive work. Accuracy and runtime
results go to `example/compare-single-vbml/outputs/`; inference results go to
`example/parallax_inference/outputs/`. Each command accepts `--output-dir PATH`.
Reports record hardware, versions, source hashes, and numerical settings.
Methods run one at a time with access to the same CPU host. They may use
different execution layouts; these are whole-machine comparisons, not
single-core measurements. Reports record worker counts, thread settings,
CPU time, and wall time. Their ratio measures average CPU utilization.
ESS per CPU-hour is `3600 * ESS / CPU_seconds`, using measured CPU time
summed over all threads and worker processes.

## Method settings

The main comparison uses one fixed configuration per method:

- **Table:** the packaged joint basis table, selected by `Model(source=...)`.
  Reports record its artifact hash, grid, and domain.
- **FFTLog:** `solver="fft"`, with the supported 4096-bin configuration.
- **VBMicrolensing:** `Tol=RelTol=1e-3`; linear limb darkening uses
  `minannuli=2`, uniform sources use 1.

These settings are compared at their measured errors, not at matched
accuracy. Additional FFT resolutions, a tighter VBM tolerance, and MulensModel
are numerical diagnostics in the reports, outside the main three-method plots.

## Accuracy reference

Lens-centered polar integration cancels the point-source singularity with the
area element. Radial source intervals are integrated analytically for uniform
brightness and numerically for limb darkening, independently of the tested
FFTs and microlensing tables:

```{literalinclude} ../example/compare-single-vbml/code/accuracy.py
:language: python
:pyobject: integral
```

The grid has 105 separations at each of eight radii from `1e-6` to 10,
including alignment, contact, and FFT approximation boundaries. References
are recomputed with quadrature tolerances `1e-9` and `1e-11`; the largest
relative change is `2.93e-12`. Uniform sources use `a1=0`, linear sources `a1=.5`.

### Reference package settings

VBM tolerances control integration and point-source switching, not the
interpolation error of its ESPL table. The smallest radius, `rho=1e-6`, lies
below its [documented table range](https://github.com/valboz/VBMicrolensing/blob/main/docs/python/SingleLenses.md).
Non-finite external-library results are counted and excluded from the error
maxima; VBM's linear-source values are missing at exact alignment. Raw data
retain the missing points. Thus a reported maximum does not
imply support for the entire sampled grid.

The supplementary MulensModel calculation uses Gould94 for uniform sources
and Yoo04 for linear limb darkening, with `gamma=2*a1/(3-a1)`. These are
small-source approximations. Neither package is silently replaced by another
method when its output is missing.

## Native batch interfaces

All methods use identical events and epochs. singlens calls
`jax.jit(model.magnification)` and synchronizes the returned array. VBM uses
`ESPLLightCurve`, receiving natural-log parameters in the order
`log(u0), log(t_e), t0, log(rho)` and physical `a1`.
Inputs and models are prepared before timing; method order is shuffled with
seed 72. Each batch has three warmups and 100 measured repeats. Reports
separate first-call cost from median warm runtime and retain every repeat.

Table interpolates the packaged bases. FFTLog constructs a transform grid
for each event evaluation; VBM uses its native tables and radial integration.
The {download}`batch benchmark <../example/compare-single-vbml/code/benchmark.py>`
contains the exact interfaces and timing loop.

(inference)=
## Inference

The [annual-parallax benchmark](benchmarks.md#inference) compares
**singlens Table + NumPyro NUTS** and **VBMicrolensing + emcee**.
singlens constructs `Model(source="linear", solver="table")`; VBM uses
`ESPLMag2` with `Tol=RelTol=1e-3` and `minannuli=2`. Both receive the same
parallax-adjusted separations at each exposure node and sample all nine
parameters, including source and blend fluxes.
Three sequential processes use noise seeds
43–45 and alternate sampler order. The physical mean, uncertainties, and
priors are fixed; only the Gaussian noise draw changes.

### Observation model

The observer is Earth's geocenter at RA $266^\circ$, Dec $-29^\circ$,
using JD TDB minus 2454222.0. The trajectory includes annual parallax from
the bundled Earth ephemeris. Independent source integration generates the
photometric mean, with lens-centered quadrature near the source and
source-centered quadrature outside it. Spatial convergence and overlap
checks accompany each reference dataset.

Nine nodes average the injection over each top-hat exposure; both fitted
models use three nodes. Exposure times and errors are fixed before fitting.
The apparent source brightness and angular radius are specified separately;
the illustrative detector and ground-network schedule are not an instrument
forecast. The {download}`experiment definition
<../example/parallax_inference/code/experiment.py>` records the physical
relations, observing assumptions, and exposure-refinement checks.

### Model and priors

The seven lens coordinates are
`(t0, log(abs(u0)/rho), log(rho*t_e), log(rho), pi_e_n, pi_e_e, logit(a1))`.
Independent Normal priors have locations
`(0, log(.5), log(.3), log(.005), 0, 0, 0)` and scales
`(.2, .7, .5, .7, .3, .3, 1)`, with `1e-6 <= rho <= 10`.
Times are in days. These are priors in the stated coordinates, conditional
on the positive impact-parameter branch.

Two additional standard-Normal coordinates map through the flux-prior
Cholesky factor to $f_s,f_b$. The NumPyro model samples all nine coordinates;
emcee evaluates the same joint Gaussian likelihood and priors using VBM
magnifications:

```{literalinclude} ../example/parallax_inference/code/posterior.py
:language: python
:pyobject: numpyro_model
```

Both samplers start from the same Table-model joint MAP and use its Hessian
for a constant affine transformation. A separate search starts from three parallax
directions for each sign of $u_0$. Its competing solutions are recorded
outside the timers; the chains compare the $u_0>0$ solution.

(flux-marginalization)=
### Flux prior

For exposure-averaged magnification $A_i$, flux has mean $f_s A_i+f_b$
and its recorded Gaussian error. Both samplers explicitly sample $f_s,f_b$
under this proper prior in total-baseline flux units:

$$
\begin{pmatrix}f_s\\f_b\end{pmatrix}
\sim\mathcal N\!\left[
\begin{pmatrix}0.8\\0.2\end{pmatrix},
\begin{pmatrix}0.0272&-0.0252\\-0.0252&0.0257\end{pmatrix}
\right].
$$

Total baseline flux therefore has mean 1 and standard deviation .05.
The prior permits signed coefficients.

Fluxes enter the Gaussian likelihood as sampled parameters; there is no
flux marginalization or conditional reconstruction in this benchmark.
ESS and stopping rules include all nine parameters. The separate
[flux marginalization example](examples/flux-marginalization.md) compares
joint and marginalized sampling under the same flux prior.

### Sampling protocol

HMC uses **NumPyro 0.22.0's `numpyro.infer.NUTS`**, which adapts the trajectory
length with the No-U-Turn criterion. Its `potential_fn` is the negative log
posterior in the shared affine coordinates; gradients come from JAX autodiff
through Table interpolation, the joint flux likelihood, and priors.
Sixteen chains run with `chain_method="parallel"`, 500 warmup steps each,
dense mass adaptation, and target acceptance 0.95. The suite configures
16 JAX CPU devices before starting each process.

**emcee 3.1.6's `EnsembleSampler`** uses its default `StretchMove`,
with two independent 32-walker ensembles and 1,000 warmup steps each.
VBM evaluates batches of walker proposals in 16 spawned processes. Each
worker uses an array binding to the installed library; its numerical core
and tables are unchanged. Separate processes isolate VBM's shared scratch state.
The common joint log likelihood and priors are evaluated in JAX without
gradients. The binding is checked against VBM's stock API; its build/load
cost in every worker, process startup, IPC, and shutdown enter the VBM total.
Both continue until estimated minimum ESS
reaches 5,000 and convergence checks pass, with caps of 12,000 NUTS draws
per chain or 32,000 emcee production steps per ensemble.

ESS uses `emcee.autocorr.integrated_time(c=5, tol=0)` on unthinned physical
parameter chains for both samplers. Every series must span at least 50
estimated autocorrelation times. NUTS additionally requires
$\widehat R<1.01$, zero divergences, and no maximum-depth saturation.
Emcee requires less than 10% change in autocorrelation estimates and
ensemble means within four combined Monte Carlo standard errors.
Walkers in one ensemble are not treated as independent chains for
$\widehat R$.

The shared model/MAP/Hessian preparation is charged to each method, along
with its compilation, warmup, and production. Data generation, diagnostics,
signed-mode searches, numerical checks, posterior selection, plots, and IO
are excluded. Timings use synchronized sequential CPU runs; they are
in-process measurements. The {download}`comparison script
<../example/parallax_inference/code/compare.py>` defines the timers and
stopping rules. Replace `run_suite` with `compare` in the reproduction
command to run one realization.

Execution layouts were chosen using a separate noise draw, seed 99.
Short NUTS runs compared sequential, parallel, and vectorized chains;
VBM batch timings compared 1–16 workers. The selected layouts were fastest
among those tested on this host. The suite sets BLAS/OpenMP thread limits
to 1; JAX/XLA remains parallel and CPU affinity is not imposed.
Measured CPU time divided by wall time is 10.8–12.2 for NUTS and 8.7–9.1 for
VBM/emcee.
Download the {download}`execution calibration
<../example/parallax_inference/outputs/execution.json>` or reproduce it with
the {download}`calibration script <../example/parallax_inference/code/execution.py>`.

### Validation and downloads

Independent source integration checks sampled truth and joint MAP epochs;
exposure refinement also checks posterior draws. JAX directional derivatives
are checked against refined finite differences in MAP/Hessian coordinates.
VBM errors, sensitivity to tighter tolerances, and agreement with the stock
API are recorded separately. These checks describe sampled states; the two
solvers retain fixed settings at their achieved accuracies.
The largest posterior-mean difference is 0.33 pooled posterior standard
deviations (22 combined Monte Carlo standard errors). Sampler convergence
does not establish agreement between these numerical targets.

Download the joint draws selected for plotting for
{download}`NUTS <../example/parallax_inference/outputs/nuts_posterior.npz>` and
{download}`emcee <../example/parallax_inference/outputs/emcee_posterior.npz>`.
The {download}`representative report <../example/parallax_inference/outputs/comparison.json>`
retains convergence summaries and parameter statistics.
The reproduction command generates complete unthinned nine-parameter
`nuts_chains.npz` and `emcee_chains.npz` archives, sample-level
`nuts_diagnostics.json`, and completion manifests in each local seed directory.
These products are excluded from Git and source distributions; they are
required for local `--aggregate-only` validation.

The {download}`three-seed summary <../example/parallax_inference/outputs/suite.json>`
records paired efficiency ranges. Individual reports are available for seeds
{download}`43 <../example/parallax_inference/outputs/seed-43/comparison.json>`,
{download}`44 <../example/parallax_inference/outputs/seed-44/comparison.json>`, and
{download}`45 <../example/parallax_inference/outputs/seed-45/comparison.json>`.
Public seed directories retain reports, convergence summaries, and selected joint draws;
the representative figures use seed 43.
Three fixed noise draws assess sensitivity to noise, not population coverage.

## FFTLog resolution

Numerical-resolution studies are separate from ordinary method selection.
The accuracy script also produces the following FFT-only diagnostic:

```{figure} ../example/compare-single-vbml/outputs/accuracy_fft_resolution.png
:alt: FFTLog convergence versus transform resolution at several source radii

Maximum sampled relative errors at fixed source radii. The dashed line marks
0.1%; small-source curves can overlap because they use the same limiting
approximation.
```

### Why accuracy levels off

FFTLog uses a fixed high-frequency damping filter. More bins refine the grid
without reducing smoothing bias near contact. A finer transform therefore
need not improve the final model once discretization error is small.

```{figure} ../example/compare-single-vbml/outputs/runtime_vs_bins.png
:alt: FFTLog runtime versus transform resolution for a fixed 100-epoch batch

Median runtime for 100 epochs per batch, with identical geometry and the
same synchronized timing procedure as the main comparison.
```

Download the {download}`resolution accuracy report
<../example/compare-single-vbml/outputs/accuracy_fft_resolution.json>` and
{download}`resolution timings <../example/compare-single-vbml/outputs/runtime_vs_bins.json>`.
The [numerical development guide](https://github.com/ShotaMiyazaki94/singlens/blob/main/tools/validation/README.md#fft-resolution-checks)
shows how to change advanced settings and check derivatives.

## FFTLog reference results

The original FFTLog-centered JSON summaries and artifact hashes are preserved in
[the reference archive](https://github.com/ShotaMiyazaki94/singlens/tree/main/example/compare-single-vbml/reference/fftlog).
Historical raw data, posterior archives, and figures remain local; they are
excluded from the public repository and source distributions. These summaries
are separate from current three-method measurements.

The maintained `solver="fft"` implementation follows
[Sugiyama (2022)](https://arxiv.org/abs/2203.06637), with the filter and endpoint
changes described in
[numerical maintenance](https://github.com/ShotaMiyazaki94/singlens/blob/main/CONTRIBUTING.md#fft-domain-and-regularization).
The archive records singlens's FFTLog implementation, rather than claiming
bitwise reproduction of the paper's code.
