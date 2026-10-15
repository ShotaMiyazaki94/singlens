# Numerical development checks

Run from the repository root after installing `.[examples,validation,test]`:

```bash
python -m pytest
python example/inference/code/inference.py --hmc
python -m tools.validation.grid_convergence
python -m tools.validation.accuracy_domain
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

