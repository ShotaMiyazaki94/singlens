# Photometry + Astrometry

Fit synthetic photometry and two-dimensional centroid positions for the
[G-dwarf event](astrometry.md#a-luminous-g-dwarf-lens).
The unresolved source and luminous lens contribute to both observables.
Astrometry measures the angular scale; annual parallax connects it to lens mass.

The injection has a $1\,M_\odot$ lens at 6 kpc and a solar-like source at
8 kpc: $t_E=42.55$ days, $\theta_E=0.5825$ mas, $u_0=0.3$, and
$\pi_E=0.0715$. Equal intrinsic F146 luminosities and foreground extinction
give $g=F_L/F_s=16/9$. The fit infers both positive fluxes independently,
without a stellar mass–luminosity constraint.

## Mock observations

Use an Earth observer and six idealized 72-day windows centered 0, 0.5, 1,
3.5, 4, and 4.5 years after peak. Seasons and peak cadence follow the scale
of the [Roman Galactic Bulge Time-Domain Survey](https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey);
dates, gaps, and Gaussian errors are experimental choices. Roman's L2 orbit
and instrument noise are outside this observing model.

| Channel | Sampling | Independent errors |
| --- | --- | --- |
| Event-season photometry | Every 12 minutes | 0.5% of baseline flux |
| Other five photometric windows | Daily, evaluated at each stated epoch | $0.005/\sqrt{120}$ in baseline-flux units, representing an ideal average of 120 exposures |
| Astrometry, all six windows | Daily | 0.10 mas per east/north coordinate |

The normalized baseline has $F_s=0.36$, $F_L=0.64$. Times are HJD
$-2450000$, with peak at 12030. The sky position is
$(\alpha,\delta)=(268.07^\circ,-29.23^\circ)$ and the microlensing parallax
vector points $30^\circ$ east of north. Epochs, measurements, and errors are
saved with the example.

(fit-the-observed-centroid)=
## Joint model

Predict flux as $F(t)=F_s A(t)+F_L$ and fit **observed centroid positions**,
including the unlensed combined-light motion:

$$
\begin{aligned}
\boldsymbol C_{\rm obs}(t)
 &=\boldsymbol C_0+\boldsymbol\mu_{\rm base}(t-t_{\rm ref})
   +\pi_{\rm base}\boldsymbol P(t)
   +\boldsymbol\delta_{\rm blend}(t),\\
\boldsymbol\delta_{\rm blend}
 &=\frac{A}{A+g}\boldsymbol\delta_s
   +\frac{g(A-1)}{(A+g)(1+g)}
     (\boldsymbol\theta_s-\boldsymbol\theta_L).
\end{aligned}
$$

Here $t-t_{\rm ref}$ is in years, $\boldsymbol P(t)$ is the full Earth
parallax factor, and $\boldsymbol\delta_s$ is the source-only shift.
The second term accounts for changing light weights. The source–lens
separation uses the light curve's annual-parallax trajectory and the
[sky rotation convention](astrometry.md#sky-coordinates).

A uniform source has externally fixed $\theta_*=0.581\,\mu$as and
$\rho=\theta_*/\theta_E$. The lens passes about 300 source radii away;
source size and limb darkening are therefore unconstrained. Over
$t_0\pm4t_E$, using a point source changes normalized flux by at most
$1.7\times10^{-6}$ and the blended centroid by $0.0004\,\mu$as at the
injection, below the assumed errors. The fit uses
`Model(source="uniform", solver="fft", parallax=...)` for both channels.

Both likelihoods use the specified Gaussian errors, without fitted jitter,
error rescaling, or a variability GP. The [Photometry + GP tutorial](tutorials/inference.ipynb)
covers correlated source variability separately.

### Likelihood and priors

The two reference positions, two proper motions, and baseline parallax are
integrated out analytically with independent zero-centered Gaussian priors
of width 10 mas, 10 mas yr$^{-1}$, and 1 mas, respectively.
`RomanJointFit.loglike()` combines this marginalized astrometric density
with the photometric likelihood:

```{literalinclude} ../example/roman_astrometry/code/fit.py
:language: python
:start-after: BEGIN JOINT LIKELIHOOD
:end-before: END JOINT LIKELIHOOD
:dedent: 4
```

Baseline coefficients are inferred, not subtracted from the input data;
their conditional posterior contributes to predicted-position uncertainty.
$\pi_{\rm base}$ is the combined-light parallax in the astrometric reference
frame. The eight event coordinates have independent Normal priors:

| Coordinate | Mean | Standard deviation |
| --- | --- | --- |
| $t_0$ (HJD $-2450000$) | 12030 | 10 days |
| $\log|u_0|$ | $\log 0.3$ | 0.8 |
| $\log t_E$ (days) | $\log 40$ | 0.7 |
| $\pi_{E,N}$, $\pi_{E,E}$ | 0 each | 0.2 each |
| $\log\theta_E$ (mas) | $\log 0.6$ | 0.7 |
| $\log F_s$, $\log F_L$ | $\log 0.5$ each | 1 each |

Logs are natural. Gaussian parallax components imply a Rayleigh prior on
$\pi_E$; positive fluxes have lognormal priors. Each sign of $u_0$ is fitted
separately, with standardized coordinates $q_k=(c_k-\mu_k)/\sigma_k$.

### NumPyro model

`sample_branch()` uses a fixed linear transformation from the MAP curvature
to improve sampling. The factor below adds the joint likelihood and the
standard-Normal prior on `q`, then cancels the auxiliary Normal density of
`z`. The transformation has a constant Jacobian, so it leaves the target
posterior unchanged.

```{literalinclude} ../example/roman_astrometry/code/fit.py
:language: python
:start-after: BEGIN JOINT NUMPYRO
:end-before: END JOINT NUMPYRO
:dedent: 4
```

NumPyro's NUTS samples the positive branch with two sequential chains, 800 warmup steps,
1200 retained draws each, and target acceptance 0.9. The
{download}`full script <../example/roman_astrometry/code/fit.py>` contains
the observable calculation, Gaussian marginalization, MAP initialization,
and `sampler.run()`.

## Recovery

The run fits 9000 fluxes and 432 two-coordinate centroids. For the preferred
$u_0>0$ branch, posterior medians and 16th–84th percentile intervals are

- $\theta_E=0.624^{+0.057}_{-0.059}$ mas;
- $\pi_E=0.087^{+0.023}_{-0.016}$;
- $M=0.88^{+0.22}_{-0.20}\,M_\odot$.

The injected angular scale and mass fall within these intervals. Mass
uncertainty includes the $\theta_E$–$\pi_E$ correlation and is conditional
on this noise realization, its priors, and assumed errors.

Four initial guesses are optimized for each sign of $u_0$. The negative
branch's best log posterior is lower by 40.9; the main figures show the
positive branch. Sampling is conditional on each sign, without assigning
equal mixture weights. The maximum-posterior difference is not an evidence ratio.

The positive branch has maximum $\widehat R=1.0032$, minimum ESS 1088,
no divergences, and no saturated tree depths. The negative branch also
passes diagnostics, with 1200 warmup steps and target acceptance 0.98.
See the {download}`summary <../example/roman_astrometry/outputs/summary.json>`
for per-branch diagnostics and intervals, or the {download}`branch overlay
<../example/roman_astrometry/outputs/branch_comparison.png>` for the
conditional alternative.

```{figure} ../example/roman_astrometry/outputs/recovery.png
:alt: Joint recovery of the G-dwarf light curve and east–north centroid-shift geometry.

Joint posterior shown with half-day photometric bins, six-day astrometric
bins in the event season, and means in later seasons; every measurement
enters the fit. The MAP **same-epoch unlensed source-plus-lens centroid**
is subtracted from data, injection, and all posterior draws. The spread
therefore includes baseline uncertainty. Bars are measurement errors;
east and north have equal angular scales.
```

```{figure} ../example/roman_astrometry/outputs/astrometry_fit.png
:alt: Astrometric observations and joint-fit residuals across all six observing seasons.

Three-day binned positions and 68%/95% latent-model intervals, with the
same MAP baseline trajectory subtracted, above individual standardized
residuals. Intervals include baseline uncertainty and exclude measurement
noise. Panels share a time axis. At the joint MAP, summed squared
standardized residuals are 9062.7 for 9000 fluxes and 818.8 for
864 astrometric coordinates.
```

```{figure} ../example/roman_astrometry/outputs/corner.png
:alt: ChainConsumer posterior corner plot with the injected parameters marked.

Posterior and injected values, with derived mass
$M=\theta_E/(\kappa\pi_E)$,
$\kappa=8.144\,{\rm mas}\,M_\odot^{-1}$, and
$\pi_E=(\pi_{E,N}^2+\pi_{E,E}^2)^{1/2}$.
The injected source distance is not imposed as a distance constraint.
```

## Reproduce the analysis

Install example dependencies and run from the repository root:

```bash
python -m pip install -e '.[examples]'
python example/roman_astrometry/code/fit.py
python example/roman_astrometry/code/plots.py --overlay
```

Outputs in `example/roman_astrometry/outputs/` include observations,
posterior samples, assumptions and diagnostics, and PNG/PDF figures.
Documentation builds display these saved results without running the sampler.
Download the synthetic {download}`photometry
<../example/roman_astrometry/outputs/photometry.csv>` and
{download}`astrometry <../example/roman_astrometry/outputs/astrometry.csv>`.
