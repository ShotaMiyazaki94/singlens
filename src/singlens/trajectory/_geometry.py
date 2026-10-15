"""Shared ICRS sky-plane geometry."""

import jax.numpy as jnp


def _sky_basis(ra_deg, dec_deg, dtype=None):
    """ICRS north, east, and line-of-sight unit vectors."""
    ra = jnp.deg2rad(jnp.asarray(ra_deg, dtype=dtype))
    dec = jnp.deg2rad(jnp.asarray(dec_deg, dtype=dtype))
    los = jnp.array([jnp.cos(ra) * jnp.cos(dec), jnp.sin(ra) * jnp.cos(dec), jnp.sin(dec)])
    east = jnp.cross(jnp.array([0.0, 0.0, 1.0], dtype=los.dtype), los)
    east /= jnp.linalg.norm(east)
    north = jnp.cross(los, east)
    north /= jnp.linalg.norm(north)
    return north, east, los
