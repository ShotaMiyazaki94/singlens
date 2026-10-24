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
compares default magnifications against independent integration at 31 source
radii from `1e-6` to 10, including alignment, source contact, approximation
boundaries, and the wings. Uniform and pure linear limb darkening bracket
physical coefficients `0<=a1<=1`; the pure quadratic profile is checked
separately. References are cached by the integration code's signature.
These checks write JSON reports to `tools/validation/outputs/`; they do not
change defaults or rerun inference.

Tests cover independent source integration,
spline values and first/second derivatives, magnification gradients, parallax
references, and broad-range Bessel functions.

The astrometry check compares source-relative centroid shifts with independent
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

