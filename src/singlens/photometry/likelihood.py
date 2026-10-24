"""Gaussian marginalization of linear source and blend fluxes."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

__all__ = ["FluxMarginalization", "marginalize_flux", "nll_ulens", "nll_ulens_general"]


class FluxMarginalization(NamedTuple):
    """Normalized log likelihood and conditional Gaussian flux posterior.

    ``mean`` and ``covariance`` use coefficient order ``(f_s, f_b)`` and
    condition on the supplied magnification and measurement errors.
    """

    log_likelihood: jax.Array
    mean: jax.Array
    covariance: jax.Array


def _arrays(*values):
    arrays = tuple(map(jnp.asarray, values))
    if any(jnp.iscomplexobj(value) for value in arrays):
        raise ValueError("Flux data, design, errors, and Gaussian priors must be real.")
    dtype = jnp.result_type(*arrays, 1.0)
    return tuple(value.astype(dtype) for value in arrays)


def _validate_shapes(flux, design, mean, covariance):
    if flux.ndim != 1 or not flux.size:
        raise ValueError("observed_flux must be a nonempty 1D array.")
    if design.shape != (flux.size, 2):
        raise ValueError("The design matrix must have shape (n, 2).")
    if mean.shape != (2,) or covariance.shape != (2, 2):
        raise ValueError("prior_mean and prior_covariance must have shapes (2,) and (2, 2).")


def _symmetric_finite(matrix):
    tolerance = 32 * jnp.finfo(matrix.dtype).eps * jnp.max(jnp.abs(matrix))
    return jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.abs(matrix - matrix.T) <= tolerance)


def _marginalize(flux, design, noise, mean, covariance, *, diagonal):
    """Prior-whitened Gaussian integral; noise is SD or a dense covariance."""
    _validate_shapes(flux, design, mean, covariance)
    prior_chol = jnp.linalg.cholesky(covariance)
    valid = (
        jnp.all(jnp.isfinite(flux))
        & jnp.all(jnp.isfinite(design))
        & jnp.all(jnp.isfinite(mean))
        & _symmetric_finite(covariance)
        & jnp.all(jnp.isfinite(prior_chol))
    )
    residual = flux - design @ mean
    scaled_design = design @ prior_chol
    if diagonal:
        if noise.ndim != 0 and noise.shape != flux.shape:
            raise ValueError("flux_err must be scalar or have the same shape as observed_flux.")
        noise = jnp.broadcast_to(noise, flux.shape)
        valid &= jnp.all(jnp.isfinite(noise) & (noise > 0))
        residual = residual / noise
        scaled_design = scaled_design / noise[:, None]
        logdet_noise = 2 * jnp.sum(jnp.log(noise))
    else:
        if noise.shape != (flux.size, flux.size):
            raise ValueError("The noise covariance must have shape (n, n).")
        noise_chol = jnp.linalg.cholesky(noise)
        valid &= _symmetric_finite(noise) & jnp.all(jnp.isfinite(noise_chol))
        residual = solve_triangular(noise_chol, residual, lower=True)
        scaled_design = solve_triangular(noise_chol, scaled_design, lower=True)
        logdet_noise = 2 * jnp.sum(jnp.log(jnp.diag(noise_chol)))
    # Eliminate one whitened coefficient after projecting out its design
    # column. Forming I + Z.T @ Z instead loses the unit prior precision for
    # nearly proportional columns and diffuse priors, even in float64.
    # Pivoting bounds the projection coefficient and also handles zero columns.
    swap = jnp.sum(scaled_design[:, 1] ** 2) > jnp.sum(scaled_design[:, 0] ** 2)
    pivoted = jnp.where(swap, scaled_design[:, ::-1], scaled_design)
    first, second = pivoted.T
    norm = first @ first
    projection = (first @ second) / jnp.where(norm > 0, norm, 1.0)
    orthogonal = second - projection * first
    inverse = 1 / (1 + norm)
    saturation = norm * inverse
    schur = 1 + orthogonal @ orthogonal + projection**2 * saturation
    rhs = first @ residual
    # Project both vectors: a rounding-sized parallel component of the
    # second column must not pull its unconstrained coefficient away from
    # the prior when the data follow the first column exactly.
    orthogonal_residual = residual - first * (rhs / jnp.where(norm > 0, norm, 1.0))
    mean_second = (orthogonal @ orthogonal_residual + projection * rhs * inverse) / schur
    mean_first = rhs * inverse - projection * saturation * mean_second
    mean_white = jnp.stack((mean_first, mean_second))
    # Combine the parallel prediction analytically, avoiding cancellation
    # between two large, opposing source/blend contributions.
    remaining = (
        residual - first * ((rhs + projection * mean_second) * inverse) - orthogonal * mean_second
    )
    covariance_factor = jnp.array(
        [
            [jnp.sqrt(inverse), -projection * saturation / jnp.sqrt(schur)],
            [0.0, 1 / jnp.sqrt(schur)],
        ],
        dtype=flux.dtype,
    )
    mean_white = jnp.where(swap, mean_white[::-1], mean_white)
    covariance_factor = jnp.where(swap, covariance_factor[::-1], covariance_factor)
    # Residual plus prior penalty avoids subtracting two large quadratic forms.
    quadratic = remaining @ remaining + mean_white @ mean_white
    logdet = logdet_noise + jnp.log1p(norm) + jnp.log(schur)
    log_likelihood = -0.5 * (quadratic + logdet + flux.size * jnp.log(2 * jnp.pi))
    posterior_mean = mean + prior_chol @ mean_white
    factor = prior_chol @ covariance_factor
    posterior_covariance = factor @ factor.T
    return FluxMarginalization(
        *(
            jnp.where(valid, value, jnp.nan)
            for value in (log_likelihood, posterior_mean, posterior_covariance)
        )
    )


def marginalize_flux(magnification, observed_flux, flux_err, *, prior_mean, prior_covariance):
    """Integrate source/blend fluxes with an explicit Gaussian prior.

    The model is ``observed_flux = f_s*magnification + f_b + noise``.
    Supply matching nonempty 1D magnification/flux arrays and positive
    measurement standard deviations (scalar or matching array). The prior
    mean has shape ``(2,)`` and its positive-definite covariance ``(2, 2)``,
    both in ``(f_s, f_b)`` order. Gaussian priors allow negative fluxes.

    Returns the normalized log likelihood and conditional flux mean/covariance.
    Constant magnification and single observations are supported. Invalid
    numeric inputs return NaN in every field, including under JIT; shape
    mistakes raise ``ValueError``. Use ``jax.vmap`` for independent datasets.
    """
    magnification, observed_flux, flux_err, prior_mean, prior_covariance = _arrays(
        magnification, observed_flux, flux_err, prior_mean, prior_covariance
    )
    if magnification.ndim != 1 or magnification.shape != observed_flux.shape:
        raise ValueError("magnification and observed_flux must be matching 1D arrays.")
    design = jnp.column_stack((magnification, jnp.ones_like(magnification)))
    return _marginalize(
        observed_flux, design, flux_err, prior_mean, prior_covariance, diagonal=True
    )


def nll_ulens(flux, M, sigma2_obs, sigma2_fs, sigma2_fb):
    """Marginal NLL with diagonal noise and zero-mean independent flux priors.

    ``M`` has shape ``(n, 2)``. ``sigma2_obs`` is a scalar or length-n array
    of observational variances; ``sigma2_fs`` and ``sigma2_fb`` are positive
    scalar prior variances. The constant ``n/2 * log(2*pi)`` is omitted.
    """
    flux, M, sigma2_obs, sigma2_fs, sigma2_fb = _arrays(flux, M, sigma2_obs, sigma2_fs, sigma2_fb)
    if sigma2_fs.ndim or sigma2_fb.ndim:
        raise ValueError("Source and blend prior variances must be scalar.")
    prior = jnp.diag(jnp.stack((sigma2_fs, sigma2_fb)))
    result = _marginalize(
        flux, M, jnp.sqrt(sigma2_obs), jnp.zeros(2, dtype=flux.dtype), prior, diagonal=True
    )
    return -result.log_likelihood - 0.5 * flux.size * jnp.log(2 * jnp.pi)


def nll_ulens_general(flux, M, C, mu, Lambda):
    """Marginal NLL with dense noise covariance and a Gaussian flux prior.

    ``M``, ``C``, ``mu`` and ``Lambda`` have shapes ``(n, 2)``, ``(n, n)``,
    ``(2,)`` and ``(2, 2)``. Both covariance matrices must be symmetric and
    positive definite. The constant ``n/2 * log(2*pi)`` is omitted.
    """
    flux, M, C, mu, Lambda = _arrays(flux, M, C, mu, Lambda)
    result = _marginalize(flux, M, C, mu, Lambda, diagonal=False)
    return -result.log_likelihood - 0.5 * flux.size * jnp.log(2 * jnp.pi)
