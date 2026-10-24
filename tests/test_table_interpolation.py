"""Exact shared-radius spline contraction and batched automatic derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens.lensing.basis_table import BasisTable
from singlens.lensing.point import pspl


class _CustomBases:
    """Arbitrary functions exercise both supported spline representations."""

    def basis(self, u, rho):
        x, y = jnp.log(rho), u / rho
        return tuple(
            jnp.atleast_1d(v)
            for v in (jnp.exp(0.3 * x + 0.01 * y**2), jnp.exp(-0.2 * x + 0.03 * y))
        )

    def joint_basis(self, u, rho):
        x, y = jnp.log(rho), jnp.log1p((u / rho) ** 2)
        q = 1 + (u / rho) ** 2
        baseline = pspl(jnp.hypot(u, rho))
        return tuple(
            jnp.atleast_1d(v)
            for v in (
                baseline * jnp.exp((0.3 * x + 0.02 * y**2) / q),
                baseline * jnp.exp((-0.2 * x + 0.03 * y) / q),
                1 / (u**2 + 2) + (-0.2 + 0.03 * x + 0.02 * y**2) / q,
                1 / (u**2 + 2) + (0.05 - 0.01 * x + 0.04 * y) / q,
            )
        )


@pytest.fixture(scope="module", params=[False, True], ids=["photometry", "joint"])
def custom_table(request):
    return BasisTable.build(
        _CustomBases(),
        np.geomspace(0.01, 1, 8),
        np.linspace(0, 40, 16),
        astrometry=request.param,
    )


@pytest.mark.parametrize("count", [256, 257])
def test_shared_radius_contraction_matches_small_batches_and_gradients(custom_table, count):
    base_positions = jnp.linspace(0, 0.5, count)
    theta = jnp.array([1.0, 0.1, 0.43])
    channels = 2 if custom_table.has_astrometry else 1

    def observable(positions, p):
        if custom_table.has_astrometry:
            return jnp.stack(custom_table.joint(positions, p[1], a1=p[2]), axis=-1)
        return custom_table.A(positions, p[1], a1=p[2])[..., None]

    def evaluate(p, split):
        positions = base_positions * p[0]
        if split:
            return jnp.concatenate(
                [observable(chunk, p) for chunk in jnp.split(positions, [127, 254])]
            )
        if count == 256:
            positions = positions.reshape(16, 16)
        return observable(positions, p).reshape(-1, channels)

    def full(p):
        return evaluate(p, False)

    def split(p):
        return evaluate(p, True)

    np.testing.assert_allclose(jax.jit(full)(theta), split(theta), rtol=4e-14, atol=2e-14)
    np.testing.assert_allclose(
        jax.jit(jax.jacfwd(full))(theta),
        jax.jacfwd(split)(theta),
        rtol=3e-12,
        atol=3e-12,
    )
    weights = jnp.linspace(0.2, 1.7, count * channels).reshape(count, channels)
    actual_value, actual_gradient = jax.jit(
        jax.value_and_grad(lambda p: jnp.sum(weights * full(p)))
    )(theta)
    expected_value, expected_gradient = jax.value_and_grad(lambda p: jnp.sum(weights * split(p)))(
        theta
    )
    np.testing.assert_allclose(actual_value, expected_value, rtol=4e-14, atol=2e-14)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=3e-12, atol=3e-12)
