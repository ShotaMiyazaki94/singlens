"""Scientific checks for the optional Roman/GP inference example."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.stats import norm

import singlens as sl

pytest.importorskip("tinygp")
pytest.importorskip("numpyro")
pytest.importorskip("corner")

from tinygp.solvers import QuasisepSolver

from example.inference.code.inference import (
    MAG_ZEROPOINT,
    NAMES,
    PARALLAX,
    PRIOR_LOC,
    PRIOR_SCALE,
    Injection,
    build_gp,
    event_parameters,
    gp_log_probability,
    mean_flux,
    posterior_magnitudes,
    synthetic_event,
    unpack,
)


def source_covariance(times, params):
    lag = jnp.abs(times[:, None] - times[None, :])
    omega = 2 * jnp.pi / params["gp_period"]
    return (
        params["gp_sigma"] ** 2
        * jnp.exp(-lag / params["gp_scale"])
        * (jnp.cos(omega * lag) + jnp.sin(omega * lag) / (omega * params["gp_scale"]))
    )


def test_gp_likelihood_and_gradient_against_dense_covariance():
    times = jnp.array([0.0, 0.013, 0.07, 0.5, 0.9, 2.0])
    error = jnp.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    residual = jnp.array([0.02, -0.01, 0.02, 0.05, -0.08, 0.01])

    def quasisep(theta):
        return build_gp(
            times,
            error,
            {
                "gp_sigma": jnp.exp(theta[0]),
                "gp_scale": jnp.exp(theta[1]),
                "gp_period": jnp.exp(theta[2]),
            },
        ).log_probability(residual)

    def dense(theta):
        sigma, scale, period = jnp.exp(theta)
        covariance = source_covariance(
            times, dict(gp_sigma=sigma, gp_scale=scale, gp_period=period)
        ) + jnp.diag(error**2)
        return -0.5 * (
            residual @ jnp.linalg.solve(covariance, residual)
            + jnp.linalg.slogdet(covariance)[1]
            + len(times) * jnp.log(2 * jnp.pi)
        )

    theta = jnp.log(jnp.array([0.08, 0.7, 0.5]))
    actual = jax.value_and_grad(quasisep)(theta)
    expected = jax.value_and_grad(dense)(theta)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-10, atol=1e-10)
    gp = build_gp(times, error, {"gp_sigma": 0.08, "gp_scale": 0.7, "gp_period": 0.5})
    assert isinstance(gp.solver, QuasisepSolver)
    assert isinstance(gp.condition(residual, diag=0.0).gp.solver, QuasisepSolver)


def test_magnified_source_gp_likelihood_and_gradient():
    """Compare the scaled-source likelihood with the full observed-flux covariance."""
    model = sl.Model(
        source="linear", fft=sl.FFTConfig(n_fft=128), parallax=sl.AnnualParallax(**PARALLAX)
    )
    times = jnp.array([-3.0, -0.1, 0.0, 0.1, 0.2, 4.0])
    theta = jnp.asarray(PRIOR_LOC).at[0].set(0.2)
    params = unpack(theta)
    error = jnp.array([0.1, 0.5, 0.6, 0.5, 0.4, 0.1])
    flux = mean_flux(model, times, params) + jnp.array([0.05, -0.2, -0.1, 0.2, 0.1, 0.04])

    def evaluate(x, *, dense):
        p = unpack(x)
        magnification = model.magnification(times, event_parameters(p))
        prior = 0.5 * jnp.sum(((x - PRIOR_LOC) / PRIOR_SCALE) ** 2)
        if not dense:
            return gp_log_probability(magnification, p, times, flux, error) - prior
        covariance = magnification[:, None] * source_covariance(times, p) * magnification[
            None, :
        ] + jnp.diag(error**2)
        residual = flux - p["f_s"] * magnification - p["f_b"]
        return (
            -0.5
            * (
                residual @ jnp.linalg.solve(covariance, residual)
                + jnp.linalg.slogdet(covariance)[1]
                + len(times) * jnp.log(2 * jnp.pi)
            )
            - prior
        )

    actual = jax.jit(jax.value_and_grad(lambda x: evaluate(x, dense=False)))(theta)
    expected = jax.jit(jax.value_and_grad(lambda x: evaluate(x, dense=True)))(theta)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=1e-8, atol=1e-8)


def test_roman_injection_sampling_and_noise():
    model, data = synthetic_event()
    assert model.parallax is not None
    assert model.source == "linear"
    assert data.truth["a1"] == 0.5
    assert data.truth["gp_sigma"] / data.truth["f_s"] == 0.01
    assert data.truth["gp_period"] == 5.0
    assert data.truth["gp_scale"] == 20.0
    assert data.truth["pi_e_n"] == 0.2
    assert data.truth["pi_e_e"] == -0.1
    assert data.times.shape == (8640,)
    np.testing.assert_allclose(np.diff(data.times), 12 / 1440, rtol=1e-10)
    np.testing.assert_allclose(np.asarray(data.times)[[0, -1]], [-36, 36 - 12 / 1440])
    assert np.all(np.asarray(data.error) > 0)
    white = np.asarray((data.flux - data.lens_flux - data.gp_signal) / data.error)
    assert abs(white.mean()) < 0.04
    assert abs(white.std() - 1) < 0.04
    assert abs(np.corrcoef(white[:-1], white[1:])[0, 1]) < 0.04
    assert np.corrcoef(data.gp_signal[:-1], data.gp_signal[1:])[0, 1] > 0.99


def test_posterior_band_against_dense_gaussian_mixture():
    """Marginalize two event/GP states and compare with analytic mixture quantiles."""
    model = sl.Model(
        source="linear",
        fft=sl.FFTConfig(n_fft=128),
        parallax=sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=0.0, time_origin=2458000.0),
    )
    times = jnp.array([-1.0, -0.2, 0.3, 1.2])
    error = jnp.array([0.06, 0.04, 0.05, 0.07])
    first = dict(
        t0=0.0,
        u0=0.7,
        t_e=3.0,
        rho=0.01,
        f_s=1.2,
        f_b=0.5,
        gp_sigma=0.08,
        gp_scale=0.7,
        pi_e_n=0.2,
        pi_e_e=-0.1,
        a1=0.5,
        gp_period=0.5,
    )
    second = dict(
        first, t_e=3.3, f_s=1.25, f_b=0.45, gp_sigma=0.12, gp_scale=1.0, a1=0.7, gp_period=0.6
    )
    lens = mean_flux(model, times, first)
    flux = lens + jnp.array([0.02, -0.03, 0.04, -0.01])
    data = Injection(times, flux, error, first, lens, jnp.zeros_like(times))
    draws = {name: np.repeat([[first[name]], [second[name]]], 2048, axis=1) for name in NAMES}
    actual = posterior_magnitudes(model, data, draws)

    means, deviations = [], []
    for params in (first, second):
        magnification = np.asarray(model.magnification(times, event_parameters(params)))
        kernel = (
            magnification[:, None]
            * np.asarray(source_covariance(times, params))
            * magnification[None, :]
        )
        covariance = kernel + np.diag(error**2)
        lens = np.asarray(mean_flux(model, times, params))
        means.append(lens + kernel @ np.linalg.solve(covariance, flux - lens))
        deviations.append(np.sqrt(np.diag(kernel - kernel @ np.linalg.solve(covariance, kernel))))
    means, deviations = np.asarray(means), np.asarray(deviations)

    # Magnitudes reverse the flux quantile order. Allow five Monte Carlo
    # standard errors, computed from the independent Gaussian-mixture density.
    for row, probability in enumerate((0.95, 0.5, 0.05)):
        for column in range(len(times)):
            mean, std = means[:, column], deviations[:, column]
            quantile = brentq(
                lambda x: norm.cdf((x - mean) / std).mean() - probability,
                (mean - 8 * std).min(),
                (mean + 8 * std).max(),
            )
            density = (norm.pdf((quantile - mean) / std) / std).mean()
            mag_error = (
                2.5
                / np.log(10)
                / quantile
                * np.sqrt(probability * (1 - probability) / draws["t0"].size)
                / density
            )
            expected = MAG_ZEROPOINT - 2.5 * np.log10(quantile)
            assert abs(actual[row, column] - expected) < 5 * mag_error
