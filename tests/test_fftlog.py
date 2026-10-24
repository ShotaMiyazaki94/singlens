"""Analytic transforms check the normalization and order-dependent grids."""

import jax.numpy as jnp
import numpy as np
import pytest

from singlens.numerics.fftlog import hankel


@pytest.mark.parametrize("order", [0, 1])
def test_gaussian_hankel_transform(order):
    r = jnp.logspace(-7, 3, 4096)
    k, transformed = hankel(r, r ** (order + 2) * jnp.exp(-(r**2) / 2), nu=1.5, N_pad=512).hankel(
        order
    )
    np.testing.assert_allclose(k, (order + 0.5) / r[::-1], rtol=1e-12)
    selected = (k >= 0.03) & (k <= 4)
    expected = k[selected] ** order * jnp.exp(-(k[selected] ** 2) / 2)
    np.testing.assert_allclose(transformed[selected], expected, rtol=1e-7, atol=1e-11)
