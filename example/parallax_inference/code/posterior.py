"""Joint nine-parameter target shared by Table/NUTS and VBM/emcee."""

import jax.numpy as jnp
import numpy as np

from example.parallax_inference.code import experiment as exp

NAMES = exp.NAMES
TRANSFORM_NAMES = (*exp.TRANSFORM_NAMES, "source_flux_white", "blend_flux_white")
PRIOR_LOC = np.r_[exp.PRIOR_LOC, 0.0, 0.0]
PRIOR_SCALE = np.r_[exp.PRIOR_SCALE, 1.0, 1.0]
FLUX_CHOLESKY = np.linalg.cholesky(exp.FLUX_PRIOR_COVARIANCE)


def pack(params):
    flux = np.array([params["f_s"], params["f_b"]])
    return np.r_[exp.pack(params), np.linalg.solve(FLUX_CHOLESKY, flux - exp.FLUX_PRIOR_MEAN)]


def unpack(theta, sign=1):
    flux = jnp.asarray(exp.FLUX_PRIOR_MEAN) + jnp.asarray(FLUX_CHOLESKY) @ theta[7:]
    return dict(exp.unpack(theta[:7], sign), f_s=flux[0], f_b=flux[1])


def log_prior(theta):
    value = exp.log_prior(theta[:7]) - 0.5 * jnp.sum(theta[7:] ** 2)
    return jnp.where(jnp.all(jnp.isfinite(theta)), value, -jnp.inf)


def log_likelihood(theta, magnification, data):
    params = unpack(theta)
    mean = params["f_s"] * magnification + params["f_b"]
    residual = (data.flux - mean) / data.error
    return -0.5 * jnp.sum(residual**2 + 2 * jnp.log(data.error) + np.log(2 * np.pi))


def negative_log_posterior(theta, model, data, sign=1):
    values = exp.magnification(model, data, theta[:7], sign)
    result = -log_likelihood(theta, values, data) - log_prior(theta)
    return jnp.where(jnp.isfinite(result), result, jnp.inf)


def residuals(theta, model, data, sign=1):
    params = unpack(theta, sign)
    values = exp.magnification(model, data, theta[:7], sign)
    return jnp.r_[
        (params["f_s"] * values + params["f_b"] - data.flux) / data.error,
        (theta - PRIOR_LOC) / PRIOR_SCALE,
    ]


def numpyro_model(model, data):
    """Sample seven lens parameters and two correlated Gaussian fluxes jointly."""
    import numpyro
    import numpyro.distributions as dist

    theta = numpyro.sample("theta", dist.Normal(PRIOR_LOC, PRIOR_SCALE).to_event(1))
    domain = jnp.isfinite(log_prior(theta))
    numpyro.factor("domain", jnp.where(domain, 0.0, -jnp.inf))
    safe_theta = jnp.where(domain, theta, jnp.asarray(PRIOR_LOC))
    params = unpack(safe_theta)
    for name, value in params.items():
        numpyro.deterministic(name, value)
    values = exp.magnification(model, data, safe_theta[:7])
    numpyro.sample(
        "observed_flux",
        dist.Normal(params["f_s"] * values + params["f_b"], data.error).to_event(1),
        obs=data.flux,
    )
