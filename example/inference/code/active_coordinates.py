"""Exact transit coordinates for the active-star finite-source posterior.

Use the trajectory in source-radius units, source crossing time, and source
flux per source radius. The original likelihood and priors are unchanged.
Arbitrary leading batch dimensions are preserved; the final axis has length 9.
"""

import jax
import jax.numpy as jnp

SAMPLING_COORDINATES = (
    "t0",
    "log_u0_over_rho",
    "log_t_star",
    "log_baseline",
    "logit_source_fraction",
    "log_source_gp_sigma",
    "log_gp_scale",
    "log_gp_period",
    "log_source_flux_over_rho",
)


def to_sampling(problem, q):
    """Map standardized original prior coordinates q into sampling coordinates."""
    c = jnp.asarray(problem.prior_mean) + jnp.asarray(q) * jnp.asarray(problem.prior_sd)
    log_fs = c[..., 3] + jax.nn.log_sigmoid(c[..., 4])
    return jnp.stack(
        (
            c[..., 0],
            c[..., 1] - c[..., 8],
            c[..., 2] + c[..., 8],
            c[..., 3],
            c[..., 4],
            log_fs + c[..., 5],
            c[..., 6],
            c[..., 7],
            log_fs - c[..., 8],
        ),
        axis=-1,
    )


def to_prior(problem, y):
    """Invert the coordinates, returning the original standardized prior q."""
    y = jnp.asarray(y)
    log_fs = y[..., 3] + jax.nn.log_sigmoid(y[..., 4])
    log_rho = log_fs - y[..., 8]
    c = jnp.stack(
        (
            y[..., 0],
            y[..., 1] + log_rho,
            y[..., 2] - log_rho,
            y[..., 3],
            y[..., 4],
            y[..., 5] - log_fs,
            y[..., 6],
            y[..., 7],
            log_rho,
        ),
        axis=-1,
    )
    return (c - jnp.asarray(problem.prior_mean)) / jnp.asarray(problem.prior_sd)


def log_prior_jacobian(problem, y):
    """Return log|det(dq/dy)|, including original prior standard deviations."""
    y = jnp.asarray(y)
    return jnp.broadcast_to(-jnp.sum(jnp.log(jnp.asarray(problem.prior_sd))), y.shape[:-1])
