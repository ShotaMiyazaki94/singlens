# Numerical development checks

Run from the repository root after installing `.[examples,validation,test]`:

```bash
python -m pytest
python example/inference/code/inference.py --hmc
python -m tools.validation.grid_convergence
python -m tools.validation.accuracy_domain
python -m tools.validation.astrometry_accuracy
```

The grid check reads the inference example's saved posterior. The domain check
compares explicit FFTLog magnifications against independent integration at 31 source
radii from `1e-6` to 10, including alignment, source contact, approximation
boundaries, and the wings. Uniform and pure linear limb darkening bracket
physical coefficients `0<=a1<=1`; the pure quadratic profile is checked
separately. References are cached by the integration code's signature.
These checks write JSON reports to `tools/validation/outputs/` without rerunning
inference. The bundled table's checks are in the joint comparison below.

Tests cover independent source integration,
spline values and first/second derivatives, magnification gradients, parallax
references, and broad-range Bessel functions.

To regenerate a joint basis table with provenance:

```bash
PYTHONPATH=src python tools/build_finite_source_table.py --output /tmp/custom-bases.npz
```

Optional `--rho-grid` and `--z-grid` read one-column grids. Generated tables
need independent validation before use; normal package evaluation uses the
bundled table without invoking construction.

The FFT astrometry check compares source-relative centroid shifts with independent
image-moment integration for uniform, pure linear limb-darkened, and pure
`mu**2` sources. Eight radii from `1e-6` to 10 cover alignment, limb contact,
approximation boundaries, and absolute separations through the weak-lensing
wings. The two independent integration geometries are cross-checked where
they overlap; wing quadrature is also checked at a higher order. Results and
reference caches are written to `tools/validation/outputs/`.

Astrometric errors are absolute, in angular Einstein-radius units, because the
shift changes sign and crosses zero. The regression threshold for these sampled
values is `1.5e-3 * min(rho, 1)`; it is not a mathematical error bound or a guarantee
of derivative accuracy. `tests/test_astrometry.py` separately checks exact limits, Cartesian
derivatives at alignment, rotations, source-size and limb-darkening derivatives,
and consistency between mixed and entirely far-field batches.

## Experimental finite-source methods

```bash
python -m tools.validation.finite_source_methods --with-table --nuts \
  --nuts-methods fftlog gould_o4 disk_32_16 basis_table
```

This compares FFTLog, fixed
concentric-disk quadrature, Gould–Yoo/moment expansions, and offline basis
interpolation. It records independent integration errors, all three first
derivatives, compilation and synchronized warm timings, and optional two-chain
NUTS diagnostics and ESS/s. See [photometry-only alternatives](../../docs/finite-source-effects.md#photometry-only-alternatives)
for domains, limitations, and reproduction options. Outputs are local numeric
diagnostics under `tools/validation/outputs/`.

## Joint photometry and astrometry

```bash
python -m tools.validation.astrometry_methods --with-tables --nuts
```

This compares FFTLog, three fixed disk quadrature
orders, and two joint-table resolutions. References independently integrate
image moments. The report includes absolute shift errors, Cartesian and
source-parameter derivatives, synchronized warm timings, and simultaneous
photometric/astrometric NUTS fits to transit and long-wing scenarios. Table
construction and compilation are reported separately. See the
[finite-source effects guide](../../docs/finite-source-effects.md#independent-accuracy-measurements) for the numerical
domain, derivative limitations, and interpretation of ESS/s.


For partial joint reruns, `--sections accuracy gradients` selects numerical
checks, `--nuts-only` selects inference, and `--resume` preserves other sections.
Run benchmarks without competing workloads. Reruns use the installed kernels;
timings can differ from the recorded photometry-only implementation.

## Joint table representation

The bundled table uses 192 radii from `1e-6` to 10 and 947 ratios from 0 to
`1e8`, dense near contact. Its four coefficient channels occupy 5.82 MB
(5.10 MB compressed). Generation uses 96 angular nodes and 64 limb nodes
per contact subinterval. Radial disk integrals are analytic; angular and
limb integrals use fixed Gauss quadrature, split where a disk crosses the lens.

For basis fluxes $F_j$ and radial shifts $\delta_j$, set
$C_j=\delta_j/u$, with its finite central limit. The stored residuals are

$$
q=1+(u/\rho)^2,\quad R=\sqrt{u^2+\rho^2},\quad
F_*=A_{\rm PS}(R),\quad C_*=(u^2+2)^{-1},
$$

$$
L_j=q\log(F_j/F_*),\qquad D_j=q(C_j-C_*),\qquad j=0,\mu.
$$

Cubic B-splines interpolate in $(\log\rho,\log q)$, then reconstruct
$F_j=F_*\exp(L_j/q)$ and $C_j=C_*+D_j/q$. The even separation coordinate
preserves central Cartesian derivatives; the analytic point-source terms
avoid cancellation of large coordinate derivatives in the small-source wings.
Outside the table domain values are NaN, except the analytic `rho=0` limit.

For canonical disk quadrature with both orders at least 16, table generation
forms the flux residual directly from analytic second moments at
`u/rho >= 1e5`. The leading omitted relative term is below `4.7e-22` there;
subtracting rounded absolute fluxes would instead introduce derivative noise.
The packaged validation report compares the regenerated artifact with the
predecessor on every grid point and midpoint and reevaluates the archived
independent value/Jacobian references. The manifest preserves the complete
previous generation and validation record; timing and NUTS measurements were
not repeated for the new artifact.

For batches sharing a source radius, evaluation contracts the radius spline
once before gathering separation coefficients. Small batches retain the direct
tensor-product sum; this choice depends only on array shape. Both compute the
same interpolation, including its derivatives.

```bash
JAX_PLATFORMS=cpu python -m tools.validation.table_evaluation
```

This compares the shared-radius contraction with the original full gather,
using identical table coefficients. It checks photometry, astrometry, and
derivatives at alignment, contact, and in the wings, then records synchronized
value, gradient, and `vmap` timings in `outputs/table_evaluation.json`.
These are evaluation timings, not sampler ESS/s measurements.

Joint gradient references use converged finite differences and central-slope
extrapolation near transits, and complex-step source integration in the wings.
Their estimated uncertainty reaches `3.1e-8 theta_E`; 96/64 quadrature reaches
reference resolution on this test grid. Tests also cover alignment derivatives,
rotations, the uniform-contact identity, and the direct quadrature tail through
`u=1e200`. Exact-contact singular slopes are excluded from numerical error maxima.

## Photometry-only variants

The conventional Gould–Yoo approximation is
$A_{\rm PS}(u)[B_0(u/\rho)-\Gamma B_1(u/\rho)]$. Orders 2 and 4 integrate
successive terms of

$$A_{\rm PS}(s)=s^{-1}+3s/8-5s^3/128+O(s^5).$$

At fixed $u/\rho$, leading relative truncation errors scale as $\rho^4$ and
$\rho^6$, plus quadrature error. Corrected orders require $u+\rho<2$ and return
NaN outside it, except the `rho=0` limit. This is a convergence domain, not an
accuracy guarantee. These kernels use quadrature and do not provide astrometry.

The two-channel photometry table interpolates in `(log(rho), u/rho)`. The
96 × 314 grid covers `1e-5 <= rho <= 10`, `0 <= u/rho <= 30`; the refined
grid has 192 × 627 points. Both use 64/32 generating quadrature. Its spline
can have a small residual slope beside alignment despite a zero exact-center
derivative. This representation differs from the bundled joint table.

## FFT resolution checks

The [three-method benchmark](../../example/compare-single-vbml/README.md)
compares fixed FFTLog, the bundled Table, and VBM. Its separate
`accuracy_fft_resolution.*` and `runtime_vs_bins.*` products retain transform
resolution diagnostics. Earlier FFTLog-centered JSON summaries are kept
in [the reference archive](../../example/compare-single-vbml/reference/fftlog/README.md)
with original artifact hashes; historical raw data, posterior archives, and
figures are excluded from the public repository.

The [inference benchmark](../../example/parallax_inference/README.md) fits a
stellar-lens event with annual parallax. singlens Table/NumPyro NUTS and VBM/emcee share independently integrated
observations, exposure averaging, priors, and three noise seeds. Source and
blend fluxes are sampled jointly under
the [stated Gaussian prior](../../docs/benchmark-methods.md#flux-prior).
Validate the joint likelihood and gradients, and check the VBM interface;
base ESS and stopping on all nine parameters.

The public FFT choice uses fixed settings. For numerical validation, import
`FFTConfig` from `singlens.lensing` and compare values and derivatives at
identical source positions:

```python
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import singlens as sl
from singlens.lensing import FFTConfig

rho = 0.01
u = rho * jnp.linspace(0.0, 3.0, 301)
values, derivatives = {}, {}
for bins in (2048, 4096, 8192):
    model = sl.Model(source="linear", solver=FFTConfig(n_fft=bins))

    def curve(radius):
        return model.magnification_at(u, rho=radius, a1=0.5)

    values[bins] = np.asarray(jax.jit(curve)(rho))
    derivatives[bins] = np.asarray(jax.jit(jax.jacfwd(curve))(rho))

relative_change = np.max(np.abs(values[4096] / values[8192] - 1))
slope_scale = np.max(np.abs(derivatives[8192]))
slope_change = np.max(np.abs(derivatives[4096] - derivatives[8192])) / slope_scale
print("Maximum relative magnification change:", relative_change)
print("Derivative change / peak derivative:", slope_change)
```

The corresponding astrometric check uses absolute angular changes:

```python
rho = 0.01
position = rho * (jnp.linspace(-3.0, 3.0, 301) + 0.2j)
shifts, slopes = {}, {}
for bins in (4096, 8192):
    model = sl.Model(source="linear", solver=FFTConfig(n_fft=bins))

    def shift_at_radius(radius):
        return model.astrometric_shift_at(position, rho=radius, a1=0.5)

    shifts[bins] = np.asarray(jax.jit(shift_at_radius)(rho))
    slopes[bins] = np.asarray(jax.jit(jax.jacfwd(shift_at_radius))(rho))

theta_e_mas = 1.2
shift_change_mas = theta_e_mas * np.max(np.abs(shifts[4096] - shifts[8192]))
slope_change = np.max(np.abs(slopes[4096] - slopes[8192])) / np.max(np.abs(slopes[8192]))
print("Maximum shift change [mas]:", shift_change_mas)
print("Derivative change / peak derivative:", slope_change)
```

Resolution sensitivity is not an independent error estimate. Contact smoothing
and the central/small-source branches need separate reference checks. The FFT
integration range is `u=1e-8..1e4`; narrowing it does not reduce cost at fixed
bin count. See the [FFT implementation notes](../../CONTRIBUTING.md#fft-domain-and-regularization).

Independent astrometric gradient tests use 8192 bins. Separate 4096-bin tests
compare autodiff to finite differences of the same numerical model. The
`1.5e-3 * min(rho,1)` astrometric threshold is a regression guard, not an error
bound. Analytic limits and covariance checks are specified in
[theory](../../docs/theory/index.md#astrometric-centroid-shift).
