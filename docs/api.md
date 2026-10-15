# API

## Event parameters → magnification

```python
A = sl.magnification(times, t0, u0, t_e, rho=rho, a1=a1)
A = sl.magnification(times, t0, u0, t_e, rho=rho, a1=a1,
                     parallax=geometry, pi_e_n=pi_e_n, pi_e_e=pi_e_e)
```

| Input | Meaning |
| --- | --- |
| `times`, `t0`, `t_e` | Observation times, peak time, positive Einstein timescale; days for parallax |
| `u0` | Signed impact parameter in Einstein-radius units |
| `rho` | Source radius; omit for PSPL, zero gives the point-source limit |
| `a1` | Physical coefficient in `I(mu)=1-a1*(1-mu)`; omit for a uniform disk |
| `parallax` | `AnnualParallax` sky position, reference epoch, and time convention |
| `pi_e_n`, `pi_e_e` | North/east microlensing parallax components; both required with geometry |
| `fft` | Optional fixed `FFTConfig`; finite sources only |

All event parameters are scalars. Results preserve the shape of `times`,
including scalar input. Use `jax.vmap` to batch events.
Missing or unused parameters raise errors; invalid numeric `t_e`, `rho`, or
`a1` yield NaN. Physical linear coefficients satisfy
`0 <= a1 <= 1`. Derivatives follow the [numerical approximation](numerics.md).

```{eval-rst}
.. autofunction:: singlens.magnification
```

## Annual parallax

```{eval-rst}
.. autoclass:: singlens.AnnualParallax
   :members: offsets
```

By default, times are HJD − 2450000. For absolute times use `time_origin=0`;
for JD TDB set `use_hjd=False`. Keep the geocentric reference epoch `t_ref`
fixed while fitting `t0`. The built-in Earth ephemeris covers 2000–2050;
queries outside its usable range return NaN.

## Models and source positions

Use `Model` to select a source profile and evaluate named event parameters.
Create the model and sky geometry before compiling its evaluation function:

```python
model = sl.Model(source="linear", parallax=geometry)
p = sl.Parameters(t0=8000., u0=.005, t_e=20., rho=.01, a1=.5,
                  pi_e_n=.2, pi_e_e=-.1)
A = jax.jit(model.magnification)(times, p)
A_at_position = model.magnification_at(.01 + .002j, rho=.01, a1=.5)
```

`source` is `point`, `uniform`, `linear`, or `mu2`. The last is the pure
`I(mu)=mu**2` profile, not a two-coefficient quadratic law. Position evaluation
uses separation or complex position directly, without applying a trajectory.

```{eval-rst}
.. autoclass:: singlens.Model
   :members: magnification, magnification_at, trajectory, flux, fit_flux
```

```{eval-rst}
.. autoclass:: singlens.Parameters
   :members: replace
```

```{eval-rst}
.. autoclass:: singlens.FFTConfig
```

## Flux and inference

For a dataset, `flux = f_s * A + f_b`. Coefficients and limb darkening can differ
between passbands while geometry is shared. `fit_flux` solves weighted least
squares at fixed magnification; this profiles fluxes rather than marginalizing
their uncertainty. Bayesian sampling should include flux priors, as in the
[inference tutorial](tutorials/inference.ipynb). Gaussian-prior marginal
likelihood helpers are available in `singlens.photometry`.

```{eval-rst}
.. autofunction:: singlens.fit_flux
```

```{eval-rst}
.. autoclass:: singlens.FluxFit
```

## Lower-level interfaces

`UniformDisk`, `LinearLimbDarkened`, and `QuadraticProfile` are the source
profile classes. They accept the same `fft=FFTConfig(...)` as `Model`;
their `.A` methods return at least 1D arrays. `pspl`,
`image_positions`, and `singlens.trajectory` provide analytic lens and
trajectory utilities.
