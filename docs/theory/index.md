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

## Flux, optimization, and probability

For one dataset, $F_i=f_s A(t_i;\theta)+f_b$ with known independent Gaussian
errors $\sigma_i$. Weighted least squares minimizes

$$\chi^2(\theta,f_s,f_b)=\sum_i\frac{[F_i^{\mathrm{obs}}-F_i]^2}{\sigma_i^2}.$$

`Model.fit_flux` solves for $f_s,f_b$ at fixed $\theta$. Optimizing the resulting
profiled residuals is convenient and differentiable. The returned flux errors
are conditional on the event parameters and do not include their uncertainty.

For Bayesian inference, the log likelihood is

$$\log p(F^{\mathrm{obs}}|\theta,f_s,f_b)=
-\tfrac12\sum_i\left[\frac{(F_i^{\mathrm{obs}}-F_i)^2}{\sigma_i^2}
+\log(2\pi\sigma_i^2)\right].$$

The HMC tutorial samples both fluxes with explicit priors. A profiled
$-\chi^2/2$ alone does **not** integrate out flux uncertainty.
`singlens.photometry` also provides analytical marginal likelihoods for
Gaussian flux priors (omitting the constant $N\log(2\pi)/2$).

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
shows posterior light curves, residuals, parameter correlations, and traces.

The example fits the `u0 > 0` branch. Other solutions, including negative
`u0` and parallax degeneracies, may require separate exploration. See [numerical limits](../numerics.md) for accuracy limits.
