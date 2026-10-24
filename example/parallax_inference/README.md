# Annual-parallax inference benchmark

Compare **singlens Table + NumPyro NUTS** and **VBMicrolensing + emcee** on
the same synthetic stellar-lens event. Both sample nine parameters:
`t0`, `u0`, `t_e`, `rho`, `pi_e_n`, `pi_e_e`, `a1`, `f_s`, and `f_b`.
Source and blend fluxes are sampled jointly under the same Gaussian prior.
See the [benchmark](../../docs/benchmarks.md#inference) for results.

singlens uses `Model(source="linear", solver="table")` and JAX gradients;
NumPyro's NUTS adapts Hamiltonian trajectories. VBM uses `ESPLMag2` with
`Tol=RelTol=1e-3` and `minannuli=2`; emcee uses its ensemble stretch move.
Both use the same parallax-adjusted exposure nodes and Gaussian flux
likelihood. This compares the finite-source engine and sampler together.

## Reproduce

Install `.[examples,validation]` and run from the repository root without
competing intensive work:

```bash
python tools/fetch_example_data.py --bootstrap
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m example.parallax_inference.code.run_suite
```

The suite repeats noise seeds 43–45 sequentially, alternating method order.
Each method may use the whole CPU: NUTS runs 16 parallel chains, and VBM
uses 16 spawned workers for emcee likelihood batches. Reports retain CPU
time, wall time, and execution settings; these are not single-core timings.
Replace `run_suite` with `compare` for one realization. Both accept
`--output-dir PATH`. The serial VBM array binding requires a C++17 compiler
and a VBMicrolensing 5.x wheel; its numerical core is unchanged and its
build/load and process startup costs are included in the VBM total.

## Experiment

A 67-day stellar-lens event has `rho=0.00458`, `u0/rho=0.46`, and
`(pi_e_n, pi_e_e)=(0.096,-0.072)`. The source has apparent I=18.5 and supplies
80% of baseline light. An idealized ground network supplies 1,199 observations
over 210 days, including five-minute peak coverage. Independent source
integration supplies the exposure-averaged injection. The setup is inspired
by [OGLE-2007-BLG-050](https://doi.org/10.1051/0004-6361/200912923);
observations, detector assumptions, and noise are synthetic.

[experiment.py](code/experiment.py) defines the data and independent reference;
[posterior.py](code/posterior.py) defines the joint density and NumPyro model;
[compare.py](code/compare.py) runs both methods. Both share a Table-based
joint MAP/Hessian initialization and target minimum estimated ESS ≥ 5,000
across all nine parameters, with additional convergence checks. Each samples
the positive-u0 solution; separate signed-mode searches assess competing
solutions. Timings include setup, compilation, warmup, and production.

Public `outputs/seed-43/` through `seed-45/` retain reports, convergence summaries, and
selected joint posterior draws. `suite.json` summarizes the runs; the top-level
figures use seed 43. Full unthinned chains, sample-level NUTS diagnostics,
completion manifests, and per-seed figures are generated locally and ignored
by Git; run the suite to regenerate them before using `--aggregate-only`.
Plot samples are selected from actual joint draws. Numerical approximation
errors and posterior differences are
reported separately from convergence.
