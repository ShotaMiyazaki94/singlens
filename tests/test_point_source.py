import jax
import jax.numpy as jnp
import numpy as np

from singlens.lensing import image_positions, pspl
from singlens.lensing.point import lens_eq, lens_eq_det_jac


def test_analytic_magnification_and_images():
    w = np.array([0.02 + 0.04j, 1.0 - 2.0j, -0.4 + 0.1j])
    u = abs(w)
    expected = (u * u + 2) / (u * np.sqrt(u * u + 4))
    np.testing.assert_allclose(pspl(w), expected, rtol=1e-14)
    z = image_positions(w)
    np.testing.assert_allclose(lens_eq(z), np.broadcast_to(w, z.shape), atol=1e-14)
    np.testing.assert_allclose(jnp.sum(1 / abs(lens_eq_det_jac(z)), axis=0), expected, rtol=1e-12)


def test_shapes_alignment_and_wings():
    assert pspl(0.1).shape == ()
    assert np.isinf(pspl(0.0))
    assert np.isnan(image_positions(0.0)).all()
    assert pspl(1e100) == 1.0
    np.testing.assert_equal(pspl([-1, 1]), np.repeat(float(pspl(1.0)), 2))
    assert pspl(np.ones((2, 3))).shape == (2, 3)


def test_jit_and_grad_match_analytic_derivative():
    u = 0.3
    expected = -8 / (u * u * (u * u + 4) ** 1.5)
    np.testing.assert_allclose(jax.jit(jax.grad(pspl))(u), expected, rtol=1e-13)
    np.testing.assert_allclose(jax.vmap(pspl)(jnp.array([0.1, 1.0])), pspl([0.1, 1.0]))
