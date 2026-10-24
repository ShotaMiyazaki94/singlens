# Flux marginalization

Source and blend fluxes enter a microlensing light curve linearly. With
Gaussian measurement errors and a Gaussian flux prior, integrate them out
analytically and sample only the lens parameters. Their posterior is the
same as in joint sampling under these priors; both flux uncertainties are
retained.

## Model

For magnifications $A_i(\theta)$, write

$$
\boldsymbol F=X(\theta)\boldsymbol\beta+\boldsymbol\epsilon,
\qquad X_i=(A_i,1),\qquad
\boldsymbol\beta=(f_s,f_b)^\mathsf T.
$$

If $\boldsymbol\epsilon\sim\mathcal N(0,C)$ and
$\boldsymbol\beta\sim\mathcal N(\mu,\Lambda)$, then

$$
p(\boldsymbol F\mid\theta)
=\mathcal N\!\left(\boldsymbol F;
X\mu,\ C+X\Lambda X^\mathsf T\right).
$$

The implementation evaluates this density through a $2\times2$ system,
including its determinant normalization. It does not construct the full
observation covariance or substitute best-fitting fluxes.
The API below uses independent errors, $C=\operatorname{diag}(\sigma_i^2)$:

```python
result = model.marginalize_flux(
    times, params, flux, flux_error,
    prior_mean=flux_prior_mean,
    prior_covariance=flux_prior_covariance,
)
numpyro.factor("photometry", result.log_likelihood)
```

See [Miyazaki & Kawahara (2025), Appendix B](https://arxiv.org/html/2510.02639v1)
for the derivation of the marginalized likelihood and conditional flux posterior.

The prior is explicit and permits signed coefficients. This analytic integral
assumes known Gaussian errors; positive-only flux priors, flux-dependent
noise, and nonlinear blended astrometric centroids need a different treatment.

## NumPyro model

The example fits a short source transit with $t_E=2.95$ hours,
$\rho=0.614$, $u_0/\rho=0.7$, and fixed $a_1=0.4$. An idealized 70.5-day
season has 8,390 observations at 12.1-minute cadence, with known Gaussian
errors and noise seed 44.

```{figure} ../../example/flux_marginalization/outputs/lightcurve.png
:alt: Synthetic finite-source light curve near the peak, with observation uncertainties and the injected model

Peak of the event used in both fits. Error bars are $1\sigma$; the line is
the exposure-averaged injected model.
```

Magnification is averaged over each exposure before calling
`singlens.photometry.marginalize_flux`:

```{literalinclude} ../../example/flux_marginalization/code/compare.py
:language: python
:pyobject: numpyro_model
```

Both formulations use `Model(source="linear", solver="table")`, the same observations,
[Gaussian flux prior](../benchmark-methods.md#flux-prior), and lens
priors. Independent Normal priors on
`(t0, log(u0/rho), log(rho*t_e), log(rho))` have locations
`(0, log(.7), log(.075), log(.6))` and scales `(.25, .6, .7, .7)`, with
times in days and `1e-6 <= rho <= 10`.
Joint sampling includes two standardized flux coordinates;
marginal sampling integrates those coordinates out. Each uses its own MAP
and Hessian for an affine sampling transform.

## Controlled comparison

Both NumPyro NUTS runs use two chains, 1000 warmup steps per chain, and target
acceptance 0.99. They require minimum lens ESS of 1000, $\widehat R<1.01$,
zero divergences, and sufficient chain length. Efficiency uses the same
four physical lens parameters: $t_0,u_0,t_E,\rho$.

```{figure} ../../example/flux_marginalization/outputs/comparison.png
:alt: Lens sampling efficiency with joint and marginalized fluxes

Efficiency includes model preparation, each fit's MAP initialization,
compilation, warmup, and production. Conditional flux reconstruction is timed
separately.
```

On an Apple M5 Pro CPU, marginalization increases minimum lens ESS/s from
**2.99 to 12.04 (4.03×)**. Joint sampling takes 343 seconds and marginal
sampling 116 seconds; conditional flux reconstruction adds 1.03 seconds.
Both pass convergence with zero divergences. All six posterior means agree
within 1.88 combined Monte Carlo standard errors.

The {download}`report <../../example/flux_marginalization/outputs/report.json>`
records timing components, convergence diagnostics, and posterior differences
with Monte Carlo uncertainties. This single event illustrates the effect of
marginalization; its speedup is not a general performance guarantee.

## Reconstruct the flux posterior

For each retained lens draw, `result.mean` and `result.covariance` give the
conditional Gaussian distribution of $(f_s,f_b)$. Draw a correlated flux pair
and keep it with that lens draw to recover the six-parameter joint posterior.
Using only the conditional means would omit part of the flux uncertainty.

The example saves these moments using the same data, exposure averaging, and
flux prior as the fit. With matching `(chain, draw)` axes:

```{literalinclude} ../../example/flux_marginalization/code/compare.py
:language: python
:pyobject: reconstruct_posterior
```

Load the saved draws and plot all six parameters with `corner`:

```python
import corner
import numpy as np

with np.load("example/flux_marginalization/outputs/marginal_draws.npz") as draws:
    samples = reconstruct_posterior(
        draws["physical"],
        draws["conditional_flux_mean"],
        draws["conditional_flux_covariance"],
    )

fig = corner.corner(
    samples.reshape(-1, 6),
    labels=[r"$t_0$ [d]", r"$u_0$", r"$t_E$ [d]", r"$\rho$", r"$f_s$", r"$f_b$"],
)
```

```{figure} ../../example/flux_marginalization/outputs/corner.png
:alt: Six-parameter corner plot comparing joint sampling with marginalized lens draws and reconstructed source and blend fluxes

Joint sampling and the reconstructed posterior, with 68% and 95% contours
and injected values in gray. For readability, this overlay shows
$\Delta t_0=t_0-t_{0,\mathrm{injected}}$ in minutes and $t_E$ in hours.
```

## Reproduce

Run from a source checkout with the example and validation dependencies:

```bash
python -m pip install -e '.[examples,validation]'
python tools/fetch_example_data.py --bootstrap
JAX_PLATFORMS=cpu MPLBACKEND=Agg python -m example.flux_marginalization.code.compare
```

The {download}`script <../../example/flux_marginalization/code/compare.py>`
saves the report, posterior draws, and PNG/PDF figures in
`example/flux_marginalization/outputs/`. Documentation builds display the
saved results without rerunning inference. The exact
{download}`measured source <../../example/flux_marginalization/outputs/measured_compare.py.txt>`
is retained with the results.
Use `--plot-only` to regenerate the figures from saved draws without rerunning
inference.
