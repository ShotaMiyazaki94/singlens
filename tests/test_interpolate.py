"""Independent spline values/derivatives and actual FSPL knot continuity."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.interpolate import CubicSpline

from singlens.lensing import UniformDisk
from singlens.numerics.interpolate import cubic_interp


def test_natural_spline_against_scipy():
    xp = jnp.array([-0.3, 0.0, 0.1, 0.7, 1.0, 1.8])
    fp = jnp.array([1.0, 1.0, 0.9, -0.2, 0.4, 0.6])
    x = jnp.linspace(float(xp[0]) + 1e-5, float(xp[-1]) - 1e-5, 301)
    reference = CubicSpline(xp, fp, bc_type="natural")

    def fn(x):
        return cubic_interp(x, xp, fp)

    for order in range(3):
        actual = jax.jit(jax.vmap(fn))(x)
        np.testing.assert_allclose(actual, reference(x, order), rtol=2e-12, atol=2e-12)
        fn = jax.grad(fn)
    np.testing.assert_allclose(cubic_interp(xp, xp, fp), fp, atol=1e-14)
    np.testing.assert_allclose(cubic_interp(jnp.array([-1.0, 2.0]), xp, fp), fp[jnp.array([0, -1])])
    assert cubic_interp(2.0, xp, fp, right=1.0) == 1.0

    # Differentiate grid values, including a flat interval and a turning point.
    def objective(values):
        return jnp.sum(cubic_interp(x, xp, values) ** 2)

    reverse = jax.jit(jax.grad(objective))(fp)
    np.testing.assert_allclose(reverse, jax.jacfwd(objective)(fp), rtol=1e-12, atol=1e-12)
    step = 1e-6
    expected = []
    for i in range(fp.size):
        delta = np.eye(fp.size)[i] * step
        plus = CubicSpline(xp, fp + delta, bc_type="natural")(x)
        minus = CubicSpline(xp, fp - delta, bc_type="natural")(x)
        expected.append(np.sum(plus**2 - minus**2) / (2 * step))
    np.testing.assert_allclose(reverse, expected, rtol=2e-8, atol=1e-8)


@pytest.mark.parametrize("rho", [1e-6, 0.01])
def test_fspl_is_c2_across_interior_knots(rho):
    model = UniformDisk()
    if rho < model.fft.rho_switch:
        knots = np.asarray(model._small_u) * rho
    else:
        knots = np.logspace(model.fft.log_u_min, model.fft.log_u_max, model.fft.n_fft)
    knot = knots[np.argmin(abs(knots / rho - 0.7))]
    # Approach the knot closely enough to isolate continuity from variation
    # of the curvature over a finite interval.
    query = jnp.array([knot * (1 - 1e-10), knot * (1 + 1e-10)])

    def fn(u):
        return model.A(u, rho)[0]

    for _ in range(2):
        fn = jax.grad(fn)
        left, right = jax.jit(jax.vmap(fn))(query)
        np.testing.assert_allclose(left, right, rtol=1e-5)


def test_configuration_type_validation():
    with pytest.raises(TypeError, match="FFTConfig"):
        UniformDisk(fft={})
