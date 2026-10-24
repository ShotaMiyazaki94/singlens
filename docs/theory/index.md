# Microlensing model

## Point lens and trajectory

With angular separation $u$ in Einstein-radius units, the total point-source
magnification is

$$A_{\mathrm{PS}}(u)=\frac{u^2+2}{u\sqrt{u^2+4}}.$$

Without parallax, the complex source position is
$w(t)=(t-t_0)/t_E+i u_0$, and $u(t)=|w(t)|$.
`Model.trajectory` returns this complex position. With annual parallax,
the model adds the projected geocentric offsets to the real and imaginary
components. Sky coordinates and reference time are static settings;
$\pi_{E,N}$ and $\pi_{E,E}$ are fitted parameters.

## Finite sources and limb darkening

For source radius $\rho$, normalized surface brightness $I(r)$ gives

$$
A_{\mathrm{FS}}(\boldsymbol u)=
\frac{\int_{|\boldsymbol r|\leq\rho}
 I(|\boldsymbol r|)A_{\mathrm{PS}}(|\boldsymbol u-\boldsymbol r|)\,d^2\boldsymbol r}
 {\int_{|\boldsymbol r|\leq\rho} I(|\boldsymbol r|)\,d^2\boldsymbol r}.
$$

The finite-source calculation follows the radial-convolution method of
[Sugiyama (2022)](https://arxiv.org/abs/2203.06637). See
[numerical accuracy](../numerics.md) for the measured errors and applicable
parameter ranges.

The linear law is $I(\mu)=1-a_1(1-\mu)$ with
$\mu=\sqrt{1-(r/\rho)^2}$. Normalizing total brightness gives

$$\Gamma=\frac{2a_1}{3-a_1},\qquad
A_{\mathrm{linear}}=(1-\Gamma)A_{\mathrm{disk}}+\Gamma A_{\mu}.$$

The coefficient $a_1$ is differentiable. `source="mu2"`
means the pure $I(\mu)=\mu^2$ profile; it is not a two-coefficient quadratic law.

For example, the profile and trajectory enter the same event-level call:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import singlens as sl

geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)


def event_magnification(times, t0, u0, t_e, rho, a1, pi_e_n, pi_e_e):
    return sl.magnification(
        times,
        t0,
        u0,
        t_e,
        rho=rho,
        a1=a1,
        parallax=geometry,
        pi_e_n=pi_e_n,
        pi_e_e=pi_e_e,
    )
```

## Astrometric centroid shift

The flux-weighted finite-source centroid follows
[Mao & Witt (1998)](https://arxiv.org/abs/astro-ph/9804045) and
[Lee et al. (2010)](https://arxiv.org/abs/1005.3021).
Here its radial convolutions are evaluated with FFTLog.

For a dark point lens with unresolved images, the point-source image centroid
relative to the lens is

$$
\boldsymbol c_{\mathrm{PS}}(\boldsymbol r)
=\frac{r^2+3}{r^2+2}\boldsymbol r,
\qquad
\delta\boldsymbol u_{\mathrm{PS}}
=\frac{\boldsymbol u}{u^2+2}.
$$

Let $S(\boldsymbol s)$ be the circular source brightness normalized to unit
unlensed flux. The finite-source flux $D$ and its image-position first moment
$\boldsymbol M$ are separate convolutions:

$$
D(\boldsymbol u)=\int S(\boldsymbol s)
 A_{\mathrm{PS}}(|\boldsymbol u-\boldsymbol s|)\,d^2\boldsymbol s,
$$

$$
\boldsymbol M(\boldsymbol u)=\int S(\boldsymbol s)
 A_{\mathrm{PS}}(|\boldsymbol u-\boldsymbol s|)
 \boldsymbol c_{\mathrm{PS}}(\boldsymbol u-\boldsymbol s)\,d^2\boldsymbol s,
\qquad
\delta\boldsymbol u_{\mathrm{FS}}=
\frac{\boldsymbol M(\boldsymbol u)}{D(\boldsymbol u)}-\boldsymbol u.
$$

Thus averaging point-source shifts alone does not give the finite-source
shift: differential magnification also changes the weighting across the
source. Circular symmetry makes $\boldsymbol M$ radial. The flux convolution
uses the order-zero Hankel kernel $J_0$, while the radial first moment uses
$J_1$, with the same source-brightness Fourier transform. Linear
limb-darkening basis weights apply to $D$ and $\boldsymbol M$ before taking
their ratio.

The shift is zero at exact alignment. For a uniform disk it also vanishes
at $u=\rho$, and its central expansion starts with
$\delta\boldsymbol u=-\boldsymbol u/(\rho^2+4)+O(u^3)$.
Its direction can therefore reverse during a source transit. These zeroes
make an absolute angular error more useful than a pointwise relative error.

`astrometric_shift` returns this dimensionless displacement in the same
trajectory frame as `Model.trajectory`. Multiplying by $\theta_E$ converts it
to an angle; a separate sky rotation maps it into astrometric coordinate
components. Annual microlensing parallax changes $\boldsymbol u(t)$, while
source proper motion/parallax and blended light require a separate observed
centroid model. See the [astrometry chapter](../astrometry.md) for light
curves, centroid tracks, sky coordinates, and blending, and
[numerical accuracy](../numerics.md#astrometric-shifts) for validation guidance.

## Flux, optimization, and probability

For one dataset, $F_i=f_s A(t_i;\theta)+f_b$ with known independent Gaussian
errors $\sigma_i$. Weighted least squares minimizes

$$\chi^2(\theta,f_s,f_b)=\sum_i\frac{[F_i^{\mathrm{obs}}-F_i]^2}{\sigma_i^2}.$$

`Model.fit_flux` solves for $f_s,f_b$ at fixed $\theta$. An optimizer can use
this fit at each trial event. Its returned flux errors assume that the event
parameters are known; a joint fit also includes their uncertainty.

For Bayesian inference, the log likelihood is

$$\log p(F^{\mathrm{obs}}|\theta,f_s,f_b)=
-\tfrac12\sum_i\left[\frac{(F_i^{\mathrm{obs}}-F_i)^2}{\sigma_i^2}
+\log(2\pi\sigma_i^2)\right].$$

In code, given measured `times`, `flux`, and `flux_error` arrays:

```python
def log_likelihood(event, f_s, f_b):
    A = event_magnification(times, **event)
    prediction = f_s * A + f_b
    variance = flux_error**2
    return -0.5 * jnp.sum((flux - prediction) ** 2 / variance + jnp.log(2 * jnp.pi * variance))
```

Here `event` is a dictionary containing the seven arguments after `times`
in `event_magnification`. Add a log prior to obtain a log posterior. The
normalization term is especially important when noise parameters are inferred.

The HMC tutorial samples both fluxes with explicit priors, so their
uncertainty is included in the joint posterior.

The Roman-like injection–recovery example models rotational variability of
the source with a damped-oscillator GP:
$F^{\mathrm{obs}}_i=A_i(f_s+g_i)+f_b+\epsilon_i$.
The lens magnifies the source variability too. With
$C_{ij}=A_i A_j K_{\mathrm{GP},ij}+\delta_{ij}\sigma_i^2$ and
$r=F^{\mathrm{obs}}-F$,

$$\log p(F^{\mathrm{obs}}|\theta,f_s,f_b,\sigma_{\mathrm{GP}},\tau_{\mathrm{GP}},P_{\mathrm{rot}})
=-\tfrac12\left[r^T C^{-1}r+\log\det C+N\log(2\pi)\right].$$

The latent GP is marginalized with tinygp. Event parameters, limb darkening,
fluxes, and GP amplitude, period, and coherence time are jointly sampled
with the stated priors. The [inference tutorial](../tutorials/inference.ipynb)
shows posterior light curves, residuals, and parameter correlations.

The example fits the `u0 > 0` branch. Other solutions, including negative
`u0` and parallax degeneracies, may require separate exploration. See [numerical limits](../numerics.md) for accuracy limits.
