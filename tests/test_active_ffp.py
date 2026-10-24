"""Independent checks of the active-star photometry and temporal GP example."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal

import singlens as sl
from singlens.lensing.concentric import ConcentricDisk

pytest.importorskip("tinygp")
pytest.importorskip("numpyro")

from example.inference.code import active_ffp as active
from example.inference.code.active_coordinates import (
    log_prior_jacobian,
    to_prior,
    to_sampling,
)
from example.inference.code.active_settings import configuration, saturation_check, white_noise


@pytest.fixture(scope="module")
def small_fit():
    return active.ActiveFFPFit(
        times=np.array([-35.0, -21.3, -7.2, -1.3, -0.1, 0.2, 0.7, 1.9, 6.2, 17.0, 34.0])
    )


def _source_covariance(times, params):
    """Closed-form covariance in source-flux units, independently of tinygp."""
    lag = jnp.abs(times[:, None] - times[None, :])
    variance = (params["baseline"] * params["source_fraction"] * params["gp_rms"]) ** 2
    decay = 1 / params["gp_scale"]
    covariance = jnp.zeros_like(lag)
    for harmonic, fraction in ((1, 0.8), (2, 0.2)):
        frequency = 2 * jnp.pi * harmonic / params["gp_period"]
        covariance += (
            fraction
            * variance
            * jnp.exp(-decay * lag)
            * (jnp.cos(frequency * lag) + decay / frequency * jnp.sin(frequency * lag))
        )
    return covariance


def _physical_event(settings):
    """Angular scales calculated in SI, independent of the adopted kappa."""
    parsec_m = 3.085677581491367e16
    lens_distance = 1000 * settings["lens_distance_kpc"] * parsec_m
    source_distance = 1000 * settings["source_distance_kpc"] * parsec_m
    theta_e = np.sqrt(
        4
        * 6.67430e-11
        * settings["lens_mass_earth"]
        * 5.972167867791379e24
        / 299792458.0**2
        * (1 / lens_distance - 1 / source_distance)
    )
    theta_star = settings["source_radius_solar"] * 6.957e8 / source_distance
    radians_to_mas = 180 / np.pi * 3600 * 1000
    return theta_e * radians_to_mas, theta_star * radians_to_mas


def test_physical_angular_scales_spot_contrast_and_detector_budget(small_fit):
    fit, s = small_fit, small_fit.settings
    theta_e, theta_star = _physical_event(s)
    np.testing.assert_allclose(s["source_theta_mas"], theta_star, rtol=2e-12)
    # 8.144 is a rounded microlensing constant; allow its rounding explicitly.
    np.testing.assert_allclose(s["theta_e_mas"], theta_e, rtol=5e-5)
    np.testing.assert_allclose(s["rho"], theta_star / theta_e, rtol=5e-5)
    np.testing.assert_allclose(s["t_e_days"], 365.25 * theta_e / s["mu_rel_masyr"], rtol=5e-5)
    assert s["lens_mass_earth"] == 0.5
    np.testing.assert_allclose(s["impact_parameter_over_rho"], 0.7)

    ratio = s["spot_to_photosphere_f146_ratio"]
    assert 0 < ratio < 1
    mean_light = 1 - s["mean_projected_spot_coverage"] * (1 - ratio)
    np.testing.assert_allclose(
        10 ** (-0.4 * (s["source_mag_ab"] - s["unspotted_source_mag_ab"])), mean_light
    )
    np.testing.assert_allclose(
        s["gp_rms"], s["projected_spot_coverage_rms"] * (1 - ratio) / mean_light
    )
    assert 0.005 < s["gp_rms"] < 0.01
    np.testing.assert_allclose(
        10 ** (-0.4 * (s["source_mag_ab"] - s["baseline_mag_ab"])), s["source_fraction"]
    )
    assert s["exposure_seconds"] < 60 * s["cadence_minutes"]
    np.testing.assert_allclose(
        s["source_counts_per_exposure"] * s["source_only_poisson_fraction"] ** 2, 1.0
    )
    flux = np.array([0.8, 1.0, 1.6])
    variance = white_noise(flux, s) ** 2
    np.testing.assert_allclose(
        variance - s["systematic_floor_flux"] ** 2 * flux**2,
        s["noise_photon_fraction"] ** 2 * flux,
    )
    assert np.all(variance > flux / s["baseline_counts_per_exposure"])
    latent = np.asarray(fit.lens_flux + fit.gp_signal)
    report = saturation_check(latent, s)
    assert report["passed"]
    np.testing.assert_allclose(report["brightest_total_mag_ab"], fit.peak_mag_ab)
    assert not saturation_check(np.array([10.0]), s)["passed"]
    for invalid in (np.array([0.0]), np.array([np.nan])):
        with pytest.raises(ValueError):
            saturation_check(invalid, s)


def test_rotational_kernel_rms_and_density_against_dense_covariance(small_fit):
    times = small_fit.times
    params = small_fit.physical(jnp.asarray(small_fit.q_truth))
    errors = jnp.linspace(0.001, 0.009, len(times))
    residual = jnp.array([0.003, -0.01, 0.002, 0.0, 0.009, 0.001, -0.008, 0.002, 0.0, 0.01, -0.004])
    covariance = np.asarray(_source_covariance(times, params))
    expected_variance = (params["baseline"] * params["source_fraction"] * params["gp_rms"]) ** 2
    np.testing.assert_allclose(np.diag(covariance), expected_variance, rtol=1e-14)
    assert np.linalg.eigvalsh(covariance).min() > 0
    gp = active.build_gp(times, errors, params)
    np.testing.assert_allclose(
        gp.log_probability(residual),
        multivariate_normal.logpdf(residual, cov=covariance + np.diag(np.asarray(errors) ** 2)),
        rtol=1e-12,
    )


def test_magnified_gp_log_density_and_all_parameter_gradients(small_fit):
    fit = small_fit
    q = jnp.asarray(fit.q_truth) + jnp.array(
        [0.01, 0.03, -0.02, 0.01, -0.1, 0.12, -0.08, 0.04, 0.06]
    )

    def dense(q):
        p = fit.physical(q)
        amp = fit.magnification(p, fit.times)
        fs = p["baseline"] * p["source_fraction"]
        residual = fit.flux - fs * amp - (p["baseline"] - fs)
        covariance = amp[:, None] * _source_covariance(fit.times, p) * amp[None, :]
        covariance += jnp.diag(fit.error**2)
        return -0.5 * (
            residual @ jnp.linalg.solve(covariance, residual)
            + jnp.linalg.slogdet(covariance)[1]
            + len(fit.times) * jnp.log(2 * jnp.pi)
        )

    actual = jax.jit(jax.value_and_grad(fit.loglike))(q)
    expected = jax.jit(jax.value_and_grad(dense))(q)
    for a, b in zip(actual, expected):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=3e-9, atol=3e-9)
    # Explicitly perturb the source fraction: both GP amplitude and blended
    # mean change, so a test at source_fraction=1 would miss a missing factor.
    assert abs(float(actual[1][4])) > 1e-4


def test_conditional_latent_flux_matches_dense_gaussian(small_fit):
    fit = small_fit
    q = jnp.asarray(fit.q_truth)
    p = fit.physical(q)
    amp = np.asarray(fit.magnification(p, fit.times))
    fs = float(p["baseline"] * p["source_fraction"])
    mean = fs * amp + float(p["baseline"]) - fs
    kernel = amp[:, None] * np.asarray(_source_covariance(fit.times, p)) * amp[None, :]
    covariance = kernel + np.diag(np.asarray(fit.error) ** 2)
    expected_mean = mean + kernel @ np.linalg.solve(covariance, np.asarray(fit.flux) - mean)
    expected_covariance = kernel - kernel @ np.linalg.solve(covariance, kernel)
    lens, _, conditional_mean = jax.jit(fit.conditional)(q, jax.random.key(19))
    np.testing.assert_allclose(lens, mean, atol=2e-14)
    np.testing.assert_allclose(conditional_mean, expected_mean, rtol=2e-12, atol=2e-12)
    keys = jax.random.split(jax.random.key(2026), 4096)
    draws = np.asarray(jax.jit(jax.vmap(lambda key: fit.conditional(q, key)[1]))(keys))
    standardized_mean = (draws.mean(0) - expected_mean) / np.sqrt(
        np.diag(expected_covariance) / len(keys)
    )
    assert np.max(np.abs(standardized_mean)) < 4.5
    np.testing.assert_allclose(draws.var(0, ddof=1), np.diag(expected_covariance), rtol=0.08)


def test_exposure_integrated_transit_against_independent_quadrature(small_fit):
    fit = small_fit
    p = fit.truth
    half_transit = p["t_e"] * np.sqrt(p["rho"] ** 2 - p["u0"] ** 2)
    times = p["t0"] + half_transit * np.array([-1.5, -1.05, -0.95, 0.0, 0.95, 1.05, 1.5])
    exposure = fit.settings["exposure_seconds"] / 86400
    nodes, weights = np.polynomial.legendre.leggauss(24)
    t = times[:, None] + exposure / 2 * nodes
    u = np.hypot((t - p["t0"]) / p["t_e"], p["u0"])
    # ConcentricDisk integrates exact uniform-disk primitives and a separately
    # resolved limb-darkening integral; it does not use the FFT/interpolator.
    reference = ConcentricDisk(n_disk=128, n_limb=96)
    expected = np.asarray(reference.A(u.ravel(), p["rho"], a1=fit.a1)).reshape(u.shape)
    expected = expected @ (weights / 2)
    actual = np.asarray(fit.magnification(p, times))
    errors = white_noise(p["source_fraction"] * expected + 1 - p["source_fraction"], fit.settings)
    assert np.max(np.abs((actual - expected) * p["source_fraction"] / errors)) < 0.02

    # rho is inferred: validate its derivative inside and outside transit,
    # excluding the mathematically singular instantaneous limb-contact point.
    nodes, weights = map(jnp.asarray, (nodes, weights / 2))
    exposed = jnp.asarray(times)[:, None] + exposure / 2 * nodes

    def direct(log_rho):
        u = jnp.hypot((exposed - p["t0"]) / p["t_e"], p["u0"])
        a = reference.A(u.ravel(), jnp.exp(log_rho), a1=fit.a1)
        return a.reshape(u.shape) @ weights

    def approximate(log_rho):
        return fit.magnification({**p, "rho": jnp.exp(log_rho)}, times)

    expected_gradient = jax.jacfwd(direct)(jnp.log(p["rho"]))
    actual_gradient = jax.jacfwd(approximate)(jnp.log(p["rho"]))
    assert np.isfinite(actual_gradient).all()
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=0.01, atol=0.002)


def test_injection_white_noise_and_full_event_saturation():
    fit = active.ActiveFFPFit()
    latent = fit.lens_flux + fit.gp_signal
    whitened = np.asarray((fit.flux - latent) / fit.error)
    assert abs(whitened.mean()) < 0.04
    assert abs(whitened.std() - 1) < 0.04
    assert abs(np.corrcoef(whitened[:-1], whitened[1:])[0, 1]) < 0.04
    assert saturation_check(np.asarray(latent), fit.settings)["passed"]
    np.testing.assert_allclose(
        np.diff(fit.times), fit.settings["cadence_minutes"] / 1440, rtol=2e-12
    )
    np.testing.assert_allclose(fit.error, white_noise(np.asarray(latent), fit.settings), rtol=1e-13)


def test_physical_parallax_is_weak_after_rectilinear_refit():
    """Bound the omitted annual curvature without granting the GP extra freedom."""
    s = configuration()
    times = np.arange(-s["season_days"] / 2, s["season_days"] / 2, s["cadence_minutes"] / 1440)
    model = sl.Model(source="linear")
    annual = sl.Model(
        source="linear", parallax=sl.AnnualParallax(ra_deg=268.07, dec_deg=-29.23, t_ref=12030.2)
    )
    initial = np.array(
        [
            0.2,
            np.log(s["u0"]),
            np.log(s["t_e_days"]),
            s["source_fraction"],
            1 - s["source_fraction"],
        ]
    )

    @jax.jit
    def predict(x):
        return model.flux(
            jnp.asarray(times),
            sl.Parameters(t0=x[0], u0=jnp.exp(x[1]), t_e=jnp.exp(x[2]), rho=s["rho"], a1=0.4),
            f_s=x[3],
            f_b=x[4],
        )

    jacobian = jax.jit(jax.jacfwd(predict))
    reference = np.asarray(predict(initial))
    error = white_noise(reference, s)
    for direction in (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4):
        event = sl.Parameters(
            t0=12030.2,
            u0=s["u0"],
            t_e=s["t_e_days"],
            rho=s["rho"],
            a1=0.4,
            pi_e_n=s["pi_e"] * np.cos(direction),
            pi_e_e=s["pi_e"] * np.sin(direction),
        )
        target = np.asarray(
            annual.flux(
                times + 12030, event, f_s=s["source_fraction"], f_b=1 - s["source_fraction"]
            )
        )
        optimum = least_squares(
            lambda x: (np.asarray(predict(x)) - target) / error,
            initial,
            jac=lambda x: np.asarray(jacobian(x)) / error[:, None],
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-9,
        )
        assert optimum.success
        assert np.sum(((np.asarray(predict(optimum.x)) - target) / error) ** 2) < 2.5


def test_sampling_coordinates_scalar_and_batched_roundtrip(small_fit):
    fit = small_fit
    offsets = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.2, 1.1, -0.7, 0.4, -1.5, 0.8, -0.2, 1.0, 0.5],
            [-0.4, -1.2, 0.6, -0.5, 1.7, -0.9, 0.6, -0.5, -0.8],
            [0.6, 0.4, 1.0, 1.1, -0.8, 0.4, 1.2, -0.7, 0.3],
        ]
    )
    q = (fit.q_truth + offsets).reshape(2, 2, 9)
    y = jax.jit(lambda q: to_sampling(fit, q))(q)
    assert y.shape == q.shape
    np.testing.assert_allclose(to_prior(fit, y), q, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(to_sampling(fit, to_prior(fit, y)), y, atol=2e-13)
    for q_scalar, y_scalar in zip(q.reshape(-1, 9), y.reshape(-1, 9)):
        np.testing.assert_allclose(to_sampling(fit, q_scalar), y_scalar, atol=2e-14)
        np.testing.assert_allclose(to_prior(fit, y_scalar), q_scalar, atol=2e-13)
        # Check the source-crossing quantities independently of the inverse.
        p = {name: float(value) for name, value in fit.physical(q_scalar).items()}
        fs = p["baseline"] * p["source_fraction"]
        np.testing.assert_allclose(np.exp(y_scalar[1]), p["u0"] / p["rho"], rtol=2e-13)
        np.testing.assert_allclose(np.exp(y_scalar[2]), p["t_e"] * p["rho"], rtol=2e-13)
        np.testing.assert_allclose(np.exp(y_scalar[5]), fs * p["gp_rms"], rtol=2e-13)
        np.testing.assert_allclose(np.exp(y_scalar[8]), fs / p["rho"], rtol=2e-13)


def test_sampling_coordinate_jacobian_and_gradient_against_autodiff(small_fit):
    fit = small_fit
    center = to_sampling(fit, jnp.asarray(fit.q_truth))
    offsets = jnp.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0.1, 0.7, -0.3, 0.04, -1.1, 0.6, -0.4, 0.2, 0.6],
            [-0.3, -1.2, 0.6, -0.03, 1.4, -0.8, 0.5, -0.7, -0.7],
        ]
    )

    def autodiff_logdet(y):
        return jnp.linalg.slogdet(jax.jacfwd(lambda x: to_prior(fit, x))(y))[1]

    for y in center + offsets:
        expected = jax.value_and_grad(autodiff_logdet)(y)
        actual = jax.value_and_grad(lambda x: log_prior_jacobian(fit, x))(y)
        for a, b in zip(actual, expected):
            assert np.isfinite(a).all()
            np.testing.assert_allclose(a, b, rtol=2e-12, atol=2e-12)
    expected = jax.vmap(autodiff_logdet)(center + offsets)
    np.testing.assert_allclose(log_prior_jacobian(fit, center + offsets), expected, atol=2e-12)
