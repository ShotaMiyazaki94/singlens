"""Analytic single point-mass lens, in Einstein-radius units."""

import jax.numpy as jnp

__all__ = ["pspl", "image_positions", "lens_eq", "lens_eq_det_jac"]


def pspl(u):
    """Point-source magnification at real separation or complex position ``u``.

    Scalars and arbitrary array shapes are supported. Exact alignment returns
    infinity, as required by the point-source model.
    """
    u = jnp.abs(jnp.asarray(u))
    # Dividing first avoids overflow for large separations.
    h = jnp.hypot(u, 2.0)
    return u / h + (2.0 / h) / u


def image_positions(w):
    """Return positive/negative parity images, shape ``(2, *w.shape)``.

    At exact alignment the images form an Einstein ring; both entries are NaN
    because two unique point images do not exist.
    """
    w = jnp.asarray(w)
    u = jnp.abs(w)
    direction = w / jnp.where(u > 0, u, 1.0)
    r_plus = 0.5 * u + 0.5 * jnp.hypot(u, 2.0)
    images = jnp.stack([direction * r_plus, -direction / r_plus])
    return jnp.where(u > 0, images, jnp.nan)


def lens_eq(z):
    """Map image ``z`` to source ``w = z - 1/conj(z)``."""
    z = jnp.asarray(z)
    return z - 1.0 / jnp.conj(z)


def lens_eq_det_jac(z):
    """Signed Jacobian determinant of the single-lens equation."""
    return 1.0 - jnp.abs(jnp.asarray(z)) ** -4
