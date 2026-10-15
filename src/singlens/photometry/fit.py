"""Named results for source/blend-flux fitting."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class FluxFit(NamedTuple):
    """Weighted fit to ``observed_flux = f_s*magnification + f_b``.

    Uncertainties use the supplied measurement errors without rescaling by
    reduced chi-squared. ``model_flux`` has the observations' shape.
    """

    f_s: jax.Array
    f_b: jax.Array
    f_s_err: jax.Array
    f_b_err: jax.Array
    chi2: jax.Array
    model_flux: jax.Array


def fit_flux(magnification, observed_flux, flux_err):
    """Fit two flux coefficients; return a differentiable ``FluxFit``.

    Magnification and observations must be matching 1D arrays with at least
    two samples. ``flux_err`` is a positive scalar or matching array. Invalid
    numeric data or a degenerate design return NaN in every result field,
    including under JIT. Shape mistakes raise ``ValueError``.
    """
    magnification, observed_flux, flux_err = map(
        jnp.asarray, (magnification, observed_flux, flux_err)
    )
    if any(jnp.iscomplexobj(value) for value in (magnification, observed_flux, flux_err)):
        raise ValueError("magnification, observed_flux, and flux_err must be real.")
    if magnification.ndim != 1 or magnification.size < 2:
        raise ValueError("magnification must be a 1D array with at least two samples.")
    if observed_flux.shape != magnification.shape:
        raise ValueError("observed_flux must have the same shape as magnification.")
    if flux_err.ndim != 0 and flux_err.shape != magnification.shape:
        raise ValueError("flux_err must be scalar or have the same shape as magnification.")
    valid = (
        jnp.all(jnp.isfinite(magnification))
        & jnp.all(jnp.isfinite(observed_flux))
        & jnp.all(jnp.isfinite(flux_err) & (flux_err > 0))
        & jnp.any(magnification != magnification[0])
    )
    safe_err = jnp.where(jnp.isfinite(flux_err) & (flux_err > 0), flux_err, 1.0)
    weights = jnp.broadcast_to(1 / safe_err**2, magnification.shape)
    sumw = jnp.sum(weights)
    mean_a = jnp.sum(weights * magnification) / sumw
    mean_f = jnp.sum(weights * observed_flux) / sumw
    da, df = magnification - mean_a, observed_flux - mean_f
    saa = jnp.sum(weights * da**2)
    f_s = jnp.sum(weights * da * df) / saa
    f_b = mean_f - f_s * mean_a
    f_s_err = jnp.sqrt(1 / saa)
    f_b_err = jnp.sqrt(1 / sumw + mean_a**2 / saa)
    chi2 = jnp.sum(weights * (df - f_s * da) ** 2)
    fitted = f_s * magnification + f_b
    return FluxFit(
        *(jnp.where(valid, value, jnp.nan) for value in (f_s, f_b, f_s_err, f_b_err, chi2, fitted))
    )
