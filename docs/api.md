# API

## Event parameters → magnification

The main entry point takes observation times and scalar event parameters.
For example, evaluate a limb-darkened annual-parallax event:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import singlens as sl

times = 8000.0 + jnp.linspace(-40.0, 40.0, 501)
geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)
A = sl.magnification(
    times,
    t0=8000.0,
    u0=0.005,
    t_e=20.0,
    rho=0.01,
    a1=0.5,
    parallax=geometry,
    pi_e_n=0.2,
    pi_e_e=-0.1,
)
```

`A.shape` is `(501,)`. Omit `rho` for a point source, or `a1` for a uniform
disk. For a static event, omit the geometry and both parallax components.

| Input | Meaning |
| --- | --- |
| `times`, `t0`, `t_e` | Observation times, peak time, positive Einstein timescale; days for parallax |
| `u0` | Signed impact parameter in Einstein-radius units |
| `rho` | Source radius in Einstein-radius units; omit or set to zero for a point source |
| `a1` | Physical coefficient in `I(mu)=1-a1*(1-mu)`; omit for a uniform disk |
| `parallax` | `AnnualParallax` sky position, reference epoch, and time convention |
| `pi_e_n`, `pi_e_e` | North/east microlensing parallax components; both required with geometry |
| `fft` | Optional fixed `FFTConfig`; finite sources only |

Results preserve the shape of `times`, including scalar input. Use `jax.vmap`
to batch events.
Missing or unused parameters raise errors; invalid numeric `t_e`, `rho`, or
`a1` yield NaN. Physical linear coefficients satisfy
`0 <= a1 <= 1`. Derivatives follow the [numerical approximation](numerics.md).

```{eval-rst}
.. autofunction:: singlens.magnification
```

(astrometric-shift-api)=
## Event parameters → astrometric shift

`astrometric_shift` accepts the same event parameters and source profiles as
`magnification`. It returns the unresolved image centroid's displacement from
the unlensed source center, in angular Einstein-radius units:

```python
shift = sl.astrometric_shift(
    times,
    t0=8000.0,
    u0=0.005,
    t_e=20.0,
    rho=0.01,
    a1=0.5,
    parallax=geometry,
    pi_e_n=0.2,
    pi_e_e=-0.1,
)
theta_e_mas = 1.2
shift_mas = theta_e_mas * shift
```

Both arrays preserve `times.shape`. The real and imaginary components are
along `tau` and `beta` in the trajectory frame; they are not RA/Dec components.
Apply the event's sky rotation separately. The model assumes a dark point
lens, unresolved images, and a circular brightness profile. Source proper
motion/parallax, an absolute sky position, and the flux and positions of
blended light must be supplied by an observation model.

The [astrometry chapter](astrometry.md) compares light curves and centroid
tracks for point and finite sources, and shows how to add a luminous lens
or an unrelated blend to the observed centroid.

JIT, derivatives, and `jax.vmap` work as for magnification. Astrometric accuracy
must be assessed in angular units, including near sign changes; see
[numerical accuracy](numerics.md#astrometric-shifts).

```{eval-rst}
.. autofunction:: singlens.astrometric_shift
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
p = sl.Parameters(t0=8000.0, u0=0.005, t_e=20.0, rho=0.01, a1=0.5, pi_e_n=0.2, pi_e_e=-0.1)
A = jax.jit(model.magnification)(times, p)
A_at_position = model.magnification_at(0.01 + 0.002j, rho=0.01, a1=0.5)
shift = jax.jit(model.astrometric_shift)(times, p)
shift_at_position = model.astrometric_shift_at(0.01 + 0.002j, rho=0.01, a1=0.5)
```

`source` is `point`, `uniform`, `linear`, or `mu2`. The last is the pure
`I(mu)=mu**2` profile, not a two-coefficient quadratic law. Position evaluation
uses separation or complex position directly, without applying a trajectory.
For `astrometric_shift_at`, a real position gives a signed real shift and a
complex position gives a complex shift in the input coordinate frame. The
shape is preserved, and exact alignment returns zero. Finite-source
astrometric kernels are prepared on the first astrometric call; evaluating
photometry alone does not prepare them.

```{eval-rst}
.. autoclass:: singlens.Model
   :members: magnification, magnification_at, astrometric_shift, astrometric_shift_at, trajectory, flux, fit_flux
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
between passbands while geometry is shared. `fit_flux` finds the best-fitting
source and blend fluxes by weighted least squares for a given magnification
curve. For Bayesian inference, sample the fluxes with the event parameters
and give them explicit priors, as in the
[inference tutorial](tutorials/inference.ipynb).

Astrometric blending also needs the blend's position: a fitted `f_b` alone
does not specify the observed centroid. See the
[blended-centroid model](astrometry.md#blending-and-the-observed-centroid).

With measured arrays `observed_flux` and `flux_error` at the same times:

```python
fit = sl.fit_flux(A, observed_flux, flux_error)
fitted_flux = fit.f_s * A + fit.f_b
chi2 = jnp.sum(((observed_flux - fitted_flux) / flux_error) ** 2)
```

`fit.chi2` gives the same weighted residual sum. Its flux uncertainties are
conditional on this event curve; a joint posterior also propagates event
and limb-darkening uncertainty.

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
trajectory utilities. `singlens.lensing.point_source_shift` evaluates the
analytic point-source displacement directly at signed real or complex positions.
