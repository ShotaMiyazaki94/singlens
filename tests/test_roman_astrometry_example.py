"""Independent checks of the illustrative luminous-lens joint observation model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import least_squares
from scipy.stats import multivariate_normal, norm

pytest.importorskip("numpyro")

from example.roman_astrometry.code import fit as roman
from singlens.trajectory.ephemeris import AU_C_DAY, load_builtin_earth_ephemeris


@pytest.fixture
def small_fit():
    return roman.RomanJointFit(
        phot_times=roman.T_REF
        + np.array([-35, -20, -8, -2, 0, 3, 12, 28, 35, 180, 365, 1280, 1461, 1644]),
        ast_times=roman.T_REF + np.array([-30, -15, 0, 20, 35, 180, 365, 1280, 1644]),
    )


def _sky_axes():
    ra, dec = np.deg2rad([roman.RA_DEG, roman.DEC_DEG])
    sight = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    east = np.cross([0.0, 0.0, 1.0], sight)
    east /= np.linalg.norm(east)
    return sight, east, np.cross(sight, east)


def _earth_state(jd):
    ephemeris = load_builtin_earth_ephemeris()
    values = np.concatenate([np.asarray(ephemeris.r), np.asarray(ephemeris.v)], axis=1)
    return np.stack([np.interp(jd, np.asarray(ephemeris.t), column) for column in values.T], -1)


def _reception_state(times):
    """Invert the HJD equation using NumPy, independently of the JAX projector."""
    sight, _, _ = _sky_axes()
    hjd = np.asarray(times) + 2450000.0
    jd = hjd.copy()
    for _ in range(8):
        jd = hjd - (_earth_state(jd)[..., :3] @ sight) * AU_C_DAY
    return _earth_state(jd)


def _source_minus_lens(fit, q, times):
    """Sky separation from physical geocentric motion and differential parallax."""
    p = fit.prior_mean + np.asarray(q) * fit.prior_sd
    theta = np.exp(p[5])
    pi_e = np.hypot(p[3], p[4])
    north_motion, east_motion = p[3:5] / pi_e
    tau = (np.asarray(times) - p[0]) / np.exp(p[2])
    beta = fit.sign * np.exp(p[1])
    _, east, north = _sky_axes()
    basis = np.column_stack([east, north])
    state = _reception_state(times)
    reference = _reception_state(roman.T_REF)
    factors = -state[..., :3] @ basis
    reference_factors = -reference[:3] @ basis
    reference_velocity = -reference[3:] @ basis
    geocentric_factors = factors - reference_factors
    geocentric_factors -= np.asarray(times - roman.T_REF)[..., None] * reference_velocity
    # pi_E follows lens-minus-source motion. The source-minus-lens separation
    # therefore has the opposite velocity and parallax -pi_rel * P(t).
    delta_e = -theta * (east_motion * tau + north_motion * beta)
    delta_n = theta * (-north_motion * tau + east_motion * beta)
    delta_e -= theta * pi_e * geocentric_factors[..., 0]
    delta_n -= theta * pi_e * geocentric_factors[..., 1]
    return delta_n + 1j * delta_e


def _disk_images(position, rho):
    """Surface integral of explicit positive/negative images, in Einstein units."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    mu = (nodes + 1) / 2
    weights *= mu  # normalized disk area: 2 mu dmu
    angle = 2 * np.pi * np.arange(96) / 96
    z = np.asarray(position)[..., None, None]
    z = z + rho * np.sqrt(1 - mu[:, None] ** 2) * np.exp(1j * angle)
    radius = np.abs(z)
    discriminant = np.sqrt(radius**2 + 4)
    amp = (radius**2 + 2) / (radius * discriminant)
    image_plus = 0.5 * (radius + discriminant) * z / radius
    image_minus = 0.5 * (radius - discriminant) * z / radius
    moment = 0.5 * (amp + 1) * image_plus + 0.5 * (amp - 1) * image_minus
    return (
        np.mean(np.sum(weights[:, None] * amp, axis=-2), axis=-1),
        np.mean(np.sum(weights[:, None] * moment, axis=-2), axis=-1),
    )


def test_marginal_gaussian_matches_dense_covariance_and_conditional():
    rng = np.random.default_rng(462)
    design = rng.normal(size=(13, 5))
    sigma = rng.uniform(0.1, 0.4, 13)
    prior_sd = np.array([3.0, 3.0, 1.5, 1.5, 0.3])
    residual = rng.normal(size=13)
    prior_covariance = np.diag(prior_sd**2)
    covariance = np.diag(sigma**2) + design @ prior_covariance @ design.T
    cross = prior_covariance @ design.T
    expected = (
        multivariate_normal.logpdf(residual, cov=covariance),
        cross @ np.linalg.solve(covariance, residual),
        prior_covariance - cross @ np.linalg.solve(covariance, cross.T),
    )
    actual = jax.jit(roman.marginalized_gaussian)(
        jnp.asarray(residual), jnp.asarray(sigma), jnp.asarray(design), jnp.asarray(prior_sd)
    )
    for value, target in zip(actual, expected):
        np.testing.assert_allclose(value, target, rtol=2e-10, atol=2e-10)


def test_hjd_parallax_factors_and_full_baseline_positions(small_fit):
    fit = small_fit
    jd = 2450000.0 + roman.T_REF + np.array([-30, 0, 95, 182, 365, 1644])
    sight, east, north = _sky_axes()
    state = _earth_state(jd)
    hjd = jd + (state[:, :3] @ sight) * AU_C_DAY - 2450000.0
    expected_factors = -state[:, :3] @ np.column_stack([east, north])
    np.testing.assert_allclose(fit.earth_factors(hjd), expected_factors, rtol=0, atol=2e-11)
    beta = np.array([8.0, -12.0, -2.4, 3.1, 0.15166666666666667])
    expected = beta[:2] + np.outer((hjd - roman.T_REF) / 365.25, beta[2:4])
    expected += beta[4] * expected_factors
    np.testing.assert_allclose(fit.baseline(hjd, beta), expected, rtol=0, atol=2e-11)
    # The injected parallax belongs to the unlensed combined-light centroid.
    np.testing.assert_allclose(fit.beta_truth[4], (0.36 / 8 + 0.64 / 6))


@pytest.mark.parametrize("sign", [-1, 1])
def test_luminous_centroid_against_explicit_images_and_sky_motion(small_fit, sign):
    fit = roman.RomanJointFit(
        u0_sign=sign, phot_times=small_fit.phot_times, ast_times=small_fit.ast_times
    )
    # A different branch must fit the same simulated observations, not a new truth.
    np.testing.assert_array_equal(fit.phot_flux, small_fit.phot_flux)
    np.testing.assert_array_equal(fit.ast_xy, small_fit.ast_xy)
    q = jnp.asarray(fit.q_truth)
    p, event = fit.unpack(q)
    theta, fs, fl = np.exp(np.asarray(p)[[5, 6, 7]])
    times = fit.ast_times
    separation = _source_minus_lens(fit, q, times)
    amp, moment = _disk_images(separation / theta, float(event.rho))
    beta = np.array([8.0, -12.0, -2.4, 3.1, 0.23])
    baseline = np.asarray(fit.baseline(times, beta))
    baseline_complex = baseline[:, 1] + 1j * baseline[:, 0]
    lens_position = baseline_complex - fs / (fs + fl) * separation
    expected_flux = fs * amp + fl
    # Image light plus lens light, using absolute positions before division.
    expected_centroid = (
        fs * (theta * moment + amp * lens_position) + fl * lens_position
    ) / expected_flux
    prediction = fit.predict(q, times, beta=beta)
    np.testing.assert_allclose(prediction["flux"], expected_flux, rtol=2e-8)
    np.testing.assert_allclose(
        prediction["centroid_mas"],
        np.column_stack([expected_centroid.imag, expected_centroid.real]),
        rtol=0,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        prediction["centroid_mas"] - prediction["lensing_shift_mas"], baseline, atol=1e-13
    )
    # Vary the inferred angular Einstein radius: the fixed stellar angular
    # radius must remain fixed, rather than being a second size measurement.
    larger_q = q.at[5].add(np.log(1.7) / fit.prior_sd[5])
    larger_p, larger_event = fit.unpack(larger_q)
    np.testing.assert_allclose(larger_event.rho * np.exp(larger_p[5]), roman.THETA_STAR_MAS)
    np.testing.assert_allclose(larger_event.rho, event.rho / 1.7)


def test_no_lens_light_and_no_lensing_limits(small_fit):
    fit = small_fit
    q = jnp.asarray(fit.q_truth)
    no_light = q.at[7].set(-jnp.inf)
    p, event = fit.unpack(no_light)
    theta = np.exp(p[5])
    separation = _source_minus_lens(fit, q, fit.ast_times)
    amp, moment = _disk_images(separation / theta, float(event.rho))
    flux, shift = fit.observables(no_light, fit.ast_times)
    expected = theta * moment / amp - separation
    np.testing.assert_allclose(flux, np.exp(p[6]) * amp, rtol=2e-8)
    np.testing.assert_allclose(
        shift, np.column_stack([expected.imag, expected.real]), rtol=0, atol=2e-8
    )

    model = fit.model

    class Unlensed:
        trajectory = model.trajectory

        @staticmethod
        def magnification_at(position, **kwargs):
            return jnp.ones_like(position.real)

        @staticmethod
        def astrometric_shift_at(position, **kwargs):
            return jnp.zeros_like(position)

    fit.model = Unlensed()
    predicted = fit.predict(q, fit.ast_times, beta=fit.beta_truth)
    np.testing.assert_allclose(predicted["flux"], 1.0)
    np.testing.assert_array_equal(predicted["lensing_shift_mas"], 0.0)
    np.testing.assert_allclose(
        predicted["centroid_mas"], fit.baseline(fit.ast_times, fit.beta_truth)
    )


def test_joint_likelihood_normalization_and_gradients(small_fit):
    fit = small_fit
    q = jnp.asarray(fit.q_truth) + jnp.array([0.01, -0.02, 0.01, 0.01, -0.02, 0.03, 0.01, -0.01])
    flux, _ = fit.observables(q, fit.phot_times)
    _, shift = fit.observables(q, fit.ast_times)
    design, sd = np.asarray(fit.ast_design), np.asarray(fit.beta_prior_sd)
    covariance = np.diag(fit.ast_sigma.ravel() ** 2) + (design * sd**2) @ design.T
    expected = norm.logpdf(fit.phot_flux, flux, fit.phot_sigma).sum()
    expected += multivariate_normal.logpdf((fit.ast_xy - shift).ravel(), cov=covariance)
    np.testing.assert_allclose(fit.loglike_jit(q), expected, rtol=1e-8, atol=1e-7)
    value, gradient = fit.objective(q)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    step = 1e-5
    for index in (3, 5, 6, 7):
        offset = jnp.zeros_like(q).at[index].set(step)
        plus = -fit.loglike_jit(q + offset) + 0.5 * jnp.sum((q + offset) ** 2)
        minus = -fit.loglike_jit(q - offset) + 0.5 * jnp.sum((q - offset) ** 2)
        np.testing.assert_allclose(
            gradient[index], (plus - minus) / (2 * step), rtol=2e-4, atol=2e-5
        )


def test_noiseless_recovery_of_angular_scale_and_independent_fluxes(small_fit):
    """Recover theta_E and both fluxes without knowing the five baseline terms."""
    fit = small_fit
    design = np.asarray(fit.ast_design)
    projection = jnp.asarray(np.eye(len(design)) - design @ np.linalg.pinv(design))
    indices = jnp.array([5, 6, 7])
    truth = jnp.asarray(fit.q_truth)

    def residual(free):
        q = truth.at[indices].set(free)
        flux, _ = fit.observables(q, fit.phot_times)
        _, shift = fit.observables(q, fit.ast_times)
        # Profiling the unknown linear baseline is exact for these equal errors.
        astrometric = projection @ (jnp.asarray(fit.ast_truth) - shift).ravel()
        return jnp.concatenate(
            [(flux - fit.phot_truth) / fit.phot_sigma, astrometric / fit.ast_sigma.ravel()]
        )

    evaluate, derivative = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    solution = least_squares(
        evaluate,
        np.asarray(truth[indices]) + [0.3, 0.2, -0.2],
        jac=derivative,
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-9,
        max_nfev=80,
    )
    assert solution.success, solution.message
    np.testing.assert_allclose(solution.x, truth[indices], rtol=0, atol=2e-6)
    assert np.linalg.norm(solution.fun) < 1e-6


def test_predictive_ignores_old_posterior_and_rejects_archive_mismatch(small_fit, tmp_path):
    """A MAP-only rerun must not silently attach another run's posterior draws."""
    fit = small_fit
    archive = tmp_path / "posterior_positive.npz"
    np.savez_compressed(
        archive,
        q=(fit.q_truth + 0.2)[None, None, :],
        beta=np.zeros((1, 1, 5)),
    )
    previous_archive = archive.read_bytes()
    branch = {"map_q": fit.q_truth.tolist()}
    roman.save_predictive(fit, branch, tmp_path, "positive")
    with np.load(tmp_path / "predictive.npz") as saved:
        assert not any(name.startswith("draw_") for name in saved.files)
        assert np.isfinite(saved["map_centroid_mas"]).all()
        np.testing.assert_array_equal(saved["phot_flux"], fit.phot_flux)
    assert archive.read_bytes() == previous_archive

    # A convergence flag alone cannot authorize an archive that was replaced
    # after sampling or copied from another experiment.
    branch["sampling"] = {"converged": True, "posterior_sha256": "0" * 64}
    with pytest.raises(ValueError, match="does not match current positive sampling record"):
        roman.save_predictive(fit, branch, tmp_path, "positive")
