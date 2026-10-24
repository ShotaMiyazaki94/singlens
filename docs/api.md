# API

## Event evaluation

Construct a `Model` with an explicit source profile and fixed geometry.
Pass fitted values in `Parameters`. Methods preserve the shape of the
observation times; use `jax.vmap` to batch events.

(event-parameters-magnification)=
### Magnification

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import singlens as sl

times = 8000.0 + jnp.linspace(-40.0, 40.0, 501)
model = sl.Model(source="linear")
params = sl.Parameters(t0=8000.0, u0=0.005, t_e=20.0, rho=0.01, a1=0.5)
A = jax.jit(model.magnification)(times, params)
dA_drho = jax.jacfwd(lambda rho: model.magnification(times, params.replace(rho=rho)))(0.01)
```

| `Model(source=...)` | Required source parameters | Brightness profile |
| --- | --- | --- |
| `"point"` | None; omit `rho` and `a1` | Point source |
| `"uniform"` | `rho`; omit `a1` | Uniform disk |
| `"linear"` | `rho`, `a1` | `I(mu)=1-a1*(1-mu)` |
| `"mu2"` | `rho`; omit `a1` | `I(mu)=mu**2` |

The `mu2` profile is a pure radial basis, not a two-coefficient quadratic law.
All events require `t0`, signed `u0`, and positive `t_e`. Separations and
source radii use Einstein-radius units; `rho=0` gives the point-source limit.
The physical linear coefficient satisfies `0 <= a1 <= 1`. Missing or unused
parameters raise errors; invalid numeric `t_e`, `rho`, or `a1` yield NaN.
Derivatives follow the [numerical approximation](numerics.md).

(astrometric-shift-api)=
(event-parameters-astrometric-shift)=
(astrometric-shift)=
### Astrometry and joint evaluation

Use the same model and parameters for the unresolved image centroid's
shift from the unlensed source center:

```python
A, shift = jax.jit(model.joint)(times, params)
shift_mas = 1.2 * shift  # Angular Einstein radius: 1.2 mas.
```

The complex shift is in Einstein-radius units along the trajectory's
`tau` and `beta` axes. `model.astrometric_shift(times, params)` evaluates
it separately. A sky prediction adds rotation, source motion/parallax, and
blended light through the [observation model](astrometry.md). The kernel
assumes a dark point lens and a circular source.

(models-and-source-positions)=
### Evaluation at source positions

Position methods bypass the event trajectory and preserve the input shape:

```python
position = jnp.array([0.01 + 0.002j, 0.04 - 0.003j])
A_at, shift_at = model.joint_at(position, rho=0.01, a1=0.5)
```

Real inputs produce signed real shifts; complex inputs retain the supplied
frame. Shifts vanish at alignment. The corresponding separate methods are
`magnification_at` and `astrometric_shift_at`.

```{eval-rst}
.. autoclass:: singlens.Model
   :members: magnification, magnification_at, astrometric_shift, astrometric_shift_at, joint, joint_at, trajectory, flux, fit_flux, marginalize_flux
```

```{eval-rst}
.. autoclass:: singlens.Parameters
   :members: replace
```

## Annual parallax

Configure fixed sky geometry on the model and fitted north/east components
on the parameters:

```python
geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)
model = sl.Model(source="linear", parallax=geometry)
params = params.replace(pi_e_n=0.2, pi_e_e=-0.1)
A, shift = model.joint(times, params)
```

Both components are required with parallax geometry and omitted for a
rectilinear model. Times default to HJD − 2450000 in days. Use
`time_origin=0` for absolute times and `use_hjd=False` for JD TDB. Keep
`t_ref` fixed while fitting `t0`. The Earth ephemeris covers 2000–2050;
queries outside its usable range return NaN.

```{eval-rst}
.. autoclass:: singlens.AnnualParallax
   :members: offsets
```

## Flux and inference

`model.flux(times, params, f_s=..., f_b=...)` predicts `f_s*A + f_b`.
At fixed event parameters, fit both fluxes by weighted least squares:

```python
observed_flux = 1.3 * A + 0.2
flux_error = jnp.full_like(times, 0.01)
fit = model.fit_flux(times, params, observed_flux, flux_error)
fitted_flux = model.flux(times, params, f_s=fit.f_s, f_b=fit.f_b)
chi2 = fit.chi2
```

Flux uncertainties are conditional on the event curve. A joint posterior
also propagates event and limb-darkening uncertainty; see the
[Photometry + GP example](tutorials/inference.ipynb).

With Gaussian flux priors and known Gaussian errors, integrate both
coefficients analytically while sampling the event parameters:

```python
marginal = model.marginalize_flux(
    times, params, observed_flux, flux_error,
    prior_mean=jnp.array([1.0, 0.0]),
    prior_covariance=jnp.eye(2),
)
log_likelihood = marginal.log_likelihood
```

Pass this normalized density to `numpyro.factor`. `mean` and `covariance`
describe the conditional Gaussian distribution of `(f_s, f_b)`; draw from
it at each event sample to recover flux uncertainty. The prior is explicit
and permits signed coefficients. Positive-only or non-Gaussian flux priors
require separate treatment.

See [Flux marginalization](examples/flux-marginalization.md) for a NumPyro
model and a comparison with joint flux sampling.

Each passband can have its
own fluxes and limb coefficient. Astrometric blending additionally requires
the [blend's position](astrometry.md#observed-centroid).

```{eval-rst}
.. autoclass:: singlens.FluxFit
```

(solver-configuration)=
## Solver selection

Uniform and linear models use a precomputed table for photometry and astrometry.
Its domain is `1e-6 <= rho <= 10`, `0 <= u/rho <= 1e8`; values outside return
NaN, and `rho=0` uses the analytic point-source limit. Point sources are
analytic; pure `mu2` sources use FFTLog.

Use `solver="table"` to select the bundled table explicitly, or `solver="fft"`
for FFTLog. Both use fixed numerical settings; the solver is fixed during a fit.
See [finite-source effects](finite-source-effects.md#photometry-and-astrometry)
for examples, supported profiles, and measured accuracy.

(lower-level-interfaces)=
### Specialized submodules

Kernel development and numerical validation use `singlens.lensing`;
advanced settings are described in the
[developer instructions](https://github.com/ShotaMiyazaki94/singlens/blob/main/CONTRIBUTING.md#numerical-maintenance).
`singlens.photometry` provides fitting and Gaussian marginal likelihoods for
supplied magnifications; `singlens.trajectory` contains trajectory utilities.
