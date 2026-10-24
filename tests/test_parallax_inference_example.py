"""Physical and likelihood invariants of the stellar-parallax example."""

from importlib import import_module

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from example.parallax_inference.code import experiment as exp
from example.parallax_inference.code import posterior as target
from example.parallax_inference.code.reference import outside_integral
from singlens import Model, Parameters


def test_stellar_configuration_has_consistent_angular_and_physical_scales():
    config, event = exp.CONFIG, exp.CONFIG["event"]
    np.testing.assert_allclose(np.hypot(event["pi_e_n"], event["pi_e_e"]), config["pi_e"])
    np.testing.assert_allclose(
        config["theta_e_mas"] ** 2, 8.144 * config["lens_mass_solar"] * config["pi_rel_mas"]
    )
    np.testing.assert_allclose(
        config["pi_rel_mas"], 1 / config["lens_distance_kpc"] - 1 / config["source_distance_kpc"]
    )
    angular_radius = (
        config["source_radius_solar"]
        * 6.957e8
        / (config["source_distance_kpc"] * 1000 * 3.085677581491367e16)
    )
    np.testing.assert_allclose(
        angular_radius * 180 / np.pi * 3600 * 1000, config["theta_star_mas"], rtol=1e-12
    )
    np.testing.assert_allclose(event["rho"], config["theta_star_mas"] / config["theta_e_mas"])
    np.testing.assert_allclose(
        event["t_e"], config["theta_e_mas"] / config["mu_geo_masyr"] * 365.25
    )
    for sign in (-1, 1):
        physical = exp.unpack(exp.pack(event), sign)
        assert np.sign(physical["u0"]) == sign
        np.testing.assert_allclose(exp.pack(physical), exp.pack(event))


def test_observing_schedule_and_noise_include_the_exposure_duration():
    times = exp.observation_times()
    assert len(times) == 1199
    assert np.all(np.diff(times) > 0)
    exposure = exp.exposure_schedule(times)
    assert np.all((exposure >= 1) & (exposure <= 60))
    np.testing.assert_allclose(np.diff(times[np.abs(times) <= 1]), 5 / 1440)
    mean, durations = np.array([1.0, 20.0]), np.array([60.0, 3.0])
    rate = (
        10
        ** (
            -0.4
            * (
                exp.CONFIG["source_magnitude"]
                - exp.CONFIG["instrument_zero_point_mag_per_electron_second"]
            )
        )
        / exp.CONFIG["source_fraction"]
    )
    counts = rate * durations
    observed_variance = (
        exp.photometric_error(mean, durations) ** 2 - (0.003 * mean) ** 2
    ) * counts**2
    np.testing.assert_allclose(observed_variance, mean * counts + 500 * durations + 400)
    assert exp.saturation_check(mean, durations)["passed"]
    assert not exp.saturation_check(mean * 1e5, durations)["passed"]
    for invalid in (-1.0, 0.0, np.nan):
        with pytest.raises(ValueError, match="finite and positive"):
            exp.photometric_error(1.0, invalid)


def test_prepared_parallax_matches_public_trajectory_and_its_derivative():
    times = np.array([-30.0, -0.2, 0.0, 0.2, 80.0])
    geometry = exp.prepare_geometry(times, exp.exposure_schedule(times))
    theta = jnp.asarray(exp.pack(exp.CONFIG["event"]))
    public = Model(source="linear", parallax=exp.parallax_geometry())

    def direct(x):
        return public.trajectory(geometry.times, Parameters(**exp.unpack(x)))

    def actual(x):
        return exp.position(geometry, exp.unpack(x))

    np.testing.assert_allclose(actual(theta), direct(theta), rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(
        jax.jacfwd(actual)(theta), jax.jacfwd(direct)(theta), rtol=1e-12, atol=1e-13
    )


def _small_problem():
    times = np.array([-30.0, -0.2, 0.0, 0.2, 80.0])
    exposure = exp.exposure_schedule(times)
    geometry = exp.prepare_geometry(times, exposure)
    params = {name: exp.CONFIG["event"][name] for name in exp.NAMES}
    model = exp.make_model()
    theta = jnp.asarray(target.pack(params))
    values = jnp.sum(
        model.magnification_at(exp.position(geometry, params), rho=params["rho"], a1=params["a1"])
        * geometry.weights,
        axis=-1,
    )
    flux = params["f_s"] * values + params["f_b"]
    data = exp.Injection(
        jnp.asarray(times),
        flux,
        jnp.asarray(exp.photometric_error(flux, exposure)),
        params,
        flux,
        jnp.asarray(exposure),
        geometry,
    )
    return model, data, theta


def test_joint_likelihood_is_finite_and_differentiable_in_nine_parameters():
    model, data, theta = _small_problem()
    value, gradient = jax.jit(
        jax.value_and_grad(lambda x: target.negative_log_posterior(x, model, data))
    )(theta)
    assert np.isfinite(value) and np.all(np.isfinite(gradient))
    assert gradient.shape == (9,)
    np.testing.assert_allclose(
        data.truth["f_s"] * exp.magnification(model, data, theta) + data.truth["f_b"],
        data.lens_flux,
        rtol=1e-13,
    )
    # A directional central difference tests the complete joint objective.
    direction = np.array([0.002, 0.01, -0.01, 0.01, 0.002, -0.003, 0.01, 0.002, -0.003])
    objective = jax.jit(
        lambda step: target.negative_log_posterior(theta + step * direction, model, data)
    )
    step = 2e-4
    finite = (objective(step) - objective(-step)) / (2 * step)
    np.testing.assert_allclose(np.dot(gradient, direction), finite, atol=5e-6, rtol=5e-4)


def test_reference_validation_rejects_changed_schedule_or_bad_numerical_checks():
    times = exp.observation_times()
    exposure = exp.exposure_schedule(times)
    checks = dict(
        saturation={"passed": True},
        injection_refinement_max_flux_error_sigma=0.0,
        fitting_exposure_rule_max_flux_error_sigma=0.0,
        overlap_max_relative_difference=0.0,
    )
    error, mean = np.ones_like(times), np.ones_like(times)
    exp._validate_reference_data(times, exposure, error, mean, checks)
    with pytest.raises(ValueError, match="schedule"):
        exp._validate_reference_data(times + 1e-6, exposure, error, mean, checks)
    with pytest.raises(ValueError, match="overlap"):
        exp._validate_reference_data(
            times, exposure, error, mean, {**checks, "overlap_max_relative_difference": 1e-3}
        )


def test_fitted_and_independent_values_follow_the_states_limb_darkening():
    pytest.importorskip("matplotlib")
    times, exposure = np.array([-0.2, 0.0, 0.2]), np.ones(3)
    geometry = exp.prepare_geometry(times, exposure, order=1)
    model = exp.make_model()
    references = []
    for a1 in (0.25, 0.75):
        params = {**exp.CONFIG["event"], "a1": a1}
        recovered = exp.unpack(exp.pack(params))
        np.testing.assert_allclose(recovered["a1"], a1, atol=1e-14)
        actual = np.asarray(
            model.magnification_at(
                exp.position(geometry, recovered), rho=recovered["rho"], a1=recovered["a1"]
            )
        )[:, 0]
        reference = exp.reference_magnification(times, exposure, params, order=1)
        np.testing.assert_allclose(actual, reference, rtol=3e-6)
        references.append(reference)
    assert np.max(np.abs(references[1] / references[0] - 1)) > 0.01


def test_documented_numpyro_model_matches_the_benchmark_target():
    pytest.importorskip("numpyro")
    from numpyro.infer.util import log_density

    model, data, theta = _small_problem()
    normalizer = np.sum(np.log(target.PRIOR_SCALE) + 0.5 * np.log(2 * np.pi))
    for offset in (
        np.zeros(9),
        np.array([0.001, 0.01, -0.02, 0.01, 0.003, 0.002, -0.2, 0.1, -0.2]),
    ):
        state = theta + offset
        density, trace = log_density(target.numpyro_model, (model, data), {}, {"theta": state})
        expected = -target.negative_log_posterior(state, model, data) - normalizer
        np.testing.assert_allclose(density, expected, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(trace["a1"]["value"], exp.unpack(state)["a1"])
    outside = theta.at[3].set(np.log(11.0))
    density, _ = log_density(target.numpyro_model, (model, data), {}, {"theta": outside})
    assert np.isneginf(density)
    assert np.isposinf(target.negative_log_posterior(outside, model, data))


@pytest.mark.parametrize("a1", [0.25, 0.5, 0.75])
def test_transformed_outside_reference_matches_adaptive_source_integration(a1):
    pytest.importorskip("matplotlib")
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    rho = exp.CONFIG["event"]["rho"]
    u = np.array([1.05, 1.1, 1.2, 1.5, 2.0, 3.0]) * rho
    adaptive = np.array([integral(value, rho, a1=a1, tolerance=1e-11) for value in u])
    coarse = outside_integral(u, rho, a1=a1, order=64)
    refined = outside_integral(u, rho, a1=a1, order=128, chunk=2)
    np.testing.assert_allclose(coarse, adaptive, rtol=1e-11)
    np.testing.assert_allclose(coarse, refined, rtol=1e-12)
    with pytest.raises(ValueError, match="u/rho"):
        outside_integral(np.array([0.99 * rho]), rho, a1=a1)


@pytest.mark.parametrize("a1", [0.0, 1.0])
def test_reference_basis_limits_and_both_spatial_switches(a1):
    pytest.importorskip("matplotlib")
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    rho = exp.CONFIG["event"]["rho"]
    ratio = np.array([1.05 - 1e-8, 1.05, 1.05 + 1e-8, 3 - 1e-8, 3, 3 + 1e-8])
    params = {**exp.CONFIG["event"], "u0": 0.0, "t_e": 1.0, "pi_e_n": 0.0, "pi_e_e": 0.0, "a1": a1}
    # One centered exposure node and this trajectory give u=times exactly.
    times, duration = ratio * rho, np.ones(len(ratio))
    coarse = exp.reference_magnification(times, duration, params, order=1, spatial_order=32)
    refined = exp.reference_magnification(times, duration, params, order=1, spatial_order=64)
    adaptive = np.array([integral(value, rho, a1=a1, tolerance=1e-11) for value in times])
    np.testing.assert_allclose(coarse, adaptive, rtol=1e-11)
    np.testing.assert_allclose(coarse, refined, rtol=1e-12)
