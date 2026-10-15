import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special

from singlens.numerics.fftlog import fftlog, g_m_vals, hankel
from singlens.numerics.special import j0, j1, j2


@pytest.mark.parametrize(
    "func,ref",
    [
        (j0, special.j0),
        (j1, special.j1),
        (j2, lambda x: special.jv(2, x)),
    ],
)
def test_bessel_against_scipy(func, ref):
    x = np.r_[0.0, np.logspace(-10, 3, 100)]
    np.testing.assert_allclose(func(x), ref(x), rtol=1e-8, atol=1e-12)
    assert np.isfinite(jax.grad(func)(0.0))
    assert func(1.0).shape == ()


def test_fftlog_gamma_ratio():
    q = np.array([0.2 + 0j, 0.2 + 2j, 0.2 + 80j, 0.2 + 300j, 0.2 + 1000j])
    expected = np.exp(special.loggamma((1 + q) / 2) - special.loggamma((1 - q) / 2))
    np.testing.assert_allclose(g_m_vals(0.0, q), expected, rtol=2e-12)
    assert g_m_vals(0.0, 1.0 + 0j) == 0
    derivative = jax.jvp(lambda z: g_m_vals(0.0, z), (jnp.asarray(q),), (jnp.ones_like(q),))[1]
    expected_derivative = (
        expected * (special.digamma((1 + q) / 2) + special.digamma((1 - q) / 2)) / 2
    )
    np.testing.assert_allclose(derivative, expected_derivative, rtol=2e-12)


def test_hankel_gaussian():
    # Integral x exp(-x²/2) J0(xy) dx = exp(-y²/2).
    x = jnp.logspace(-5, 2, 2048)
    y, value = hankel(x, x * x * jnp.exp(-x * x / 2), nu=1.1, N_pad=512).hankel(0)
    mask = (y > 0.01) & (y < 3.0)
    np.testing.assert_allclose(value[mask], jnp.exp(-(y[mask] ** 2) / 2), rtol=1e-5, atol=1e-6)


def test_fftlog_no_taper():
    x = jnp.logspace(-3, 2, 64)
    _, value = fftlog(x, jnp.exp(-x), c_window_width=0).fftlog(0)
    assert np.isfinite(value).all()
