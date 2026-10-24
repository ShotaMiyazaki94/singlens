# Microlensing model

## Lens and source

### Point lens and trajectory

In Einstein-radius units, point-source magnification at separation $u$ is

$$A_{\mathrm{PS}}(u)=\frac{u^2+2}{u\sqrt{u^2+4}}.$$

For rectilinear motion, the complex source position is
$w(t)=(t-t_0)/t_E+i u_0$, with signed impact parameter $u_0$ and
$u(t)=|w(t)|$. Annual parallax adds projected geocentric offsets to its
real and imaginary components. Sky position and reference epoch are fixed;
$\pi_{E,N}$ and $\pi_{E,E}$ are fitted parameters. See
[the API](../api.md#annual-parallax) for time conventions and evaluation.

### Finite sources and limb darkening

For radius $\rho=\theta_*/\theta_E$ and circular surface brightness $I(r)$,

$$
A_{\mathrm{FS}}(\boldsymbol u)=
\frac{\int_{|\boldsymbol r|\leq\rho}
 I(|\boldsymbol r|)A_{\mathrm{PS}}(|\boldsymbol u-\boldsymbol r|)\,d^2\boldsymbol r}
 {\int_{|\boldsymbol r|\leq\rho} I(|\boldsymbol r|)\,d^2\boldsymbol r}.
$$

The linear law is $I(\mu)=1-a_1(1-\mu)$, where
$\mu=\sqrt{1-(r/\rho)^2}$ and $0\le a_1\le1$. Normalization gives

$$\Gamma=\frac{2a_1}{3-a_1},\qquad
A_{\mathrm{linear}}=(1-\Gamma)A_{\mathrm{disk}}+\Gamma A_{\mu},$$

where $A_\mu$ uses a unit-flux pure-$\mu$ profile. `source="mu2"` instead
means $I(\mu)=\mu^2$, not a two-coefficient quadratic law.

The bundled table interpolates precomputed uniform and pure-$\mu$ bases.
The optional FFTLog backend evaluates the radial convolution using
[Sugiyama (2022)](https://arxiv.org/abs/2203.06637). See
[finite-source effects](../finite-source-effects.md) for the table's generation,
interpolation, and measured accuracy.

### Astrometric centroid shift

For a dark point lens with unresolved images, the point-source centroid
relative to the lens and its displacement from the source are

$$
\boldsymbol c_{\mathrm{PS}}(\boldsymbol r)
=\frac{r^2+3}{r^2+2}\boldsymbol r,
\qquad
\delta\boldsymbol u_{\mathrm{PS}}
=\frac{\boldsymbol u}{u^2+2}.
$$

For source brightness $S(\boldsymbol s)$ normalized to unit unlensed flux,
the flux $D$ and vector image moment $\boldsymbol M$ are

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

This flux-weighted construction follows
[Mao & Witt (1998)](https://arxiv.org/abs/astro-ph/9804045) and
[Lee et al. (2010)](https://arxiv.org/abs/1005.3021). Averaging point-source
shifts alone misses magnification's changing weights across the source.
Limb-darkening weights apply to $D$ and $\boldsymbol M$ before taking their ratio.

The shift vanishes at alignment. A uniform disk also has zero shift at
$u=\rho$, with central expansion
$\delta\boldsymbol u=-\boldsymbol u/(\rho^2+4)+O(u^3)$.
Its direction can reverse during transit, making absolute angular errors
more useful than relative errors near these zeroes.

Multiply the dimensionless shift by $\theta_E$ and rotate from the trajectory
frame to sky coordinates. Source proper motion, parallax, and blended light
enter the [observed-centroid model](../astrometry.md), separately from the
microlensing trajectory.

## Flux, optimization, and probability

### Independent measurement errors

For one dataset, $F_i=f_s A(t_i;\theta)+f_b$ and independent Gaussian errors
$\sigma_i$ give

$$\chi^2(\theta,f_s,f_b)=\sum_i\frac{[F_i^{\mathrm{obs}}-F_i]^2}{\sigma_i^2},
\qquad
\log p(F^{\mathrm{obs}}|\theta,f_s,f_b)
=-\tfrac12\left[\chi^2+\sum_i\log(2\pi\sigma_i^2)\right].$$

[Flux fitting](../api.md#flux-and-inference) solves for $f_s,f_b$ at fixed
$\theta$; its uncertainties are conditional on that event curve. A joint
posterior includes event uncertainty and priors. Retain the likelihood's
normalization when inferring noise parameters.

### Source variability

A Gaussian process $g_i$ for source variability is magnified with the source:
$F_i^{\mathrm{obs}}=A_i(f_s+g_i)+f_b+\epsilon_i$.
For kernel $K_{\mathrm{GP}}$ and hyperparameters
$\boldsymbol\eta_{\mathrm{GP}}$, define
$C_{ij}=A_iA_jK_{\mathrm{GP},ij}+\delta_{ij}\sigma_i^2$ and
$r=F^{\mathrm{obs}}-F$. Marginalizing the latent process gives

$$\log p(F^{\mathrm{obs}}|\theta,f_s,f_b,\boldsymbol\eta_{\mathrm{GP}})
=-\tfrac12\left[r^T C^{-1}r+\log\det C+N\log(2\pi)\right].$$

The [Photometry + GP example](../tutorials/inference.ipynb) fits event, flux, and
GP parameters jointly with tinygp marginalization.

The two signs of $u_0$ give the same rectilinear photometric light curve.
Annual-parallax models can have additional branches requiring separate
exploration. Check [numerical accuracy](../numerics.md) for the chosen
parameters and observational precision.
