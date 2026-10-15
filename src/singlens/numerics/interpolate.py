"""JAX-native natural cubic interpolation for fixed, increasing 1-D grids."""

import jax.numpy as jnp
from jax.lax.linalg import tridiagonal_solve


def cubic_interp(x, xp, fp, *, right=None):
    """Interpolate with C2 continuity inside the grid and constant outer values.

    Natural boundary conditions set the endpoint second derivatives to zero.
    The solve is O(N), differentiable, and linear in ``fp``; in particular,
    interpolation commutes with the limb-darkening basis mixture. Like
    ``jnp.interp``, constant extension is not generally smooth at the endpoints.
    ``xp`` and ``fp`` are 1-D arrays with at least three points.
    """
    h = jnp.diff(xp)
    secant = jnp.diff(fp) / h
    span = h[:-1] + h[1:]
    lower = jnp.concatenate((jnp.zeros(1), h[:-1] / span, jnp.zeros(1)))
    upper = jnp.concatenate((jnp.zeros(1), h[1:] / span, jnp.zeros(1)))
    diagonal = jnp.concatenate((jnp.ones(1), 2 * jnp.ones_like(span), jnp.ones(1)))
    rhs = jnp.concatenate((jnp.zeros(1), 6 * jnp.diff(secant) / span, jnp.zeros(1)))
    second = tridiagonal_solve(lower, diagonal, upper, rhs[:, None])[:, 0]

    # Clip before evaluating the polynomial to keep inactive branches finite.
    query = jnp.clip(x, xp[0], xp[-1])
    index = jnp.clip(jnp.searchsorted(xp, query, side="right") - 1, 0, xp.size - 2)
    width = h[index]
    b = (query - xp[index]) / width
    a = 1 - b
    value = (
        a * fp[index]
        + b * fp[index + 1]
        + ((a**3 - a) * second[index] + (b**3 - b) * second[index + 1]) * width**2 / 6
    )
    value = jnp.where(x < xp[0], fp[0], value)
    return jnp.where(x > xp[-1], fp[-1] if right is None else right, value)
