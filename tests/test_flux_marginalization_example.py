"""The controlled example compares identical probability targets."""

from importlib import import_module
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.stats import multivariate_normal

pytestmark = pytest.mark.reference
pytest.importorskip("numpyro")
pytest.importorskip("emcee")
pytest.importorskip("corner")
pytest.importorskip("matplotlib")
pytest.importorskip("tinygp")

log_density = import_module("numpyro.infer.util").log_density

study = import_module("example.flux_marginalization.code.compare")


@pytest.fixture(scope="module")
def case():
    model = study.experiment.make_model("table")
    theta = jnp.array([0.2, np.log(0.7), np.log(0.075), np.log(0.6)])
    times = jnp.array([-0.1, 0.08, 0.15, 0.2, 0.24, 0.3, 0.5])
    values = study.experiment.magnification(model, times, study.experiment.unpack(theta))
    errors = jnp.array([0.018, 0.02, 0.014, 0.013, 0.015, 0.022, 0.019])
    flux = 0.8 * values + 0.2 + errors * jnp.array([0.2, -0.4, 0.6, -0.5, 0.1, 0.7, -0.3])
    return SimpleNamespace(times=times, flux=flux, error=errors), model, theta, values


def model_log_prob(state, data, model, formulation, center=None, transform=None):
    size = state.size
    center = jnp.zeros(size) if center is None else jnp.asarray(center)
    transform = jnp.eye(size) if transform is None else jnp.asarray(transform)
    coordinates = study.solve_triangular(transform, state - center, lower=True)
    value, _ = log_density(
        study.numpyro_model,
        (data, model, formulation, center, transform),
        {},
        {"coordinates": coordinates},
    )
    return value


@pytest.mark.parametrize("offset", [[0.2, -0.3], [-0.8, 0.5]])
def test_normalized_joint_conditional_identity_through_executed_model(case, offset):
    data, model, theta, values = case
    conditional = study.experiment.flux_conditional(values, data.flux, data.error)
    prior_chol = np.linalg.cholesky(study.experiment.FLUX_PRIOR_COVARIANCE)
    beta = np.asarray(conditional.mean) + np.linalg.cholesky(conditional.covariance) @ offset
    white = np.linalg.solve(prior_chol, beta - study.experiment.FLUX_PRIOR_MEAN)
    joint_state = jnp.r_[theta, white]
    conditional_white = (
        multivariate_normal.logpdf(
            beta, np.asarray(conditional.mean), np.asarray(conditional.covariance)
        )
        + np.linalg.slogdet(prior_chol)[1]
    )
    marginal = float(model_log_prob(theta, data, model, "marginal"))
    joint = float(model_log_prob(joint_state, data, model, "joint"))
    np.testing.assert_allclose(joint - conditional_white, marginal, rtol=0, atol=3e-10)
    np.testing.assert_allclose(
        joint,
        -study.negative_log_posterior(joint_state, data, model, "joint"),
        rtol=0,
        atol=3e-10,
    )
    np.testing.assert_allclose(
        marginal,
        -study.negative_log_posterior(theta, data, model, "marginal"),
        rtol=0,
        atol=3e-10,
    )


@pytest.mark.parametrize("formulation", ["joint", "marginal"])
def test_affine_model_keeps_exact_prior_jacobian(case, formulation):
    data, model, theta, _ = case
    state = jnp.r_[theta, jnp.array([0.15, -0.4])] if formulation == "joint" else theta
    dimension = len(state)
    center = np.linspace(-0.1, 0.2, dimension)
    factor = np.diag(np.linspace(0.3, 1.1, dimension))
    factor += np.tril(np.full((dimension, dimension), 0.04), k=-1)
    actual = model_log_prob(state, data, model, formulation, center, factor)
    expected = -study.negative_log_posterior(state, data, model, formulation)
    expected += np.linalg.slogdet(factor)[1]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=3e-10)


def test_strongly_scaled_affine_prior_is_valid_lower_triangular(case):
    from numpyro import handlers

    data, model, theta, _ = case
    center = jnp.r_[theta, jnp.zeros(2)]
    factor = np.diag([1e-5, 0.01, 0.005, 0.02, 0.04, 0.003])
    factor += np.tril(np.full((6, 6), 0.013), k=-1)
    traced = handlers.trace(handlers.seed(study.numpyro_model, jax.random.key(11))).get_trace(
        data, model, "joint", center, jnp.asarray(factor)
    )
    distribution = traced["coordinates"]["fn"]
    assert np.all(np.triu(np.asarray(distribution.scale_tril), k=1) == 0)
    distribution.validate_args(strict=True)


def test_marginal_matches_independent_observation_space_gaussian(case):
    data, model, theta, values = case
    design = np.column_stack((values, np.ones(len(values))))
    covariance = np.diag(np.asarray(data.error) ** 2)
    covariance += design @ study.experiment.FLUX_PRIOR_COVARIANCE @ design.T
    likelihood = multivariate_normal.logpdf(
        data.flux, design @ study.experiment.FLUX_PRIOR_MEAN, covariance
    )
    location, scale = map(np.asarray, study.prior_parameters("marginal"))
    prior = multivariate_normal.logpdf(theta, location, np.diag(scale**2))
    actual = study.negative_log_posterior(theta, data, model, "marginal")
    np.testing.assert_allclose(actual, -likelihood - prior, rtol=0, atol=3e-10)


@pytest.mark.parametrize("formulation", ["joint", "marginal"])
def test_outside_table_domain_has_zero_density_and_finite_gradient(case, formulation):
    data, model, theta, _ = case
    theta = theta.at[3].set(jnp.log(11.0))
    state = jnp.r_[theta, jnp.array([0.15, -0.4])] if formulation == "joint" else theta

    def objective(x):
        return study.negative_log_posterior(x, data, model, formulation)

    actual, gradient = jax.value_and_grad(objective)(state)
    model_value, model_gradient = jax.value_and_grad(
        lambda x: model_log_prob(x, data, model, formulation)
    )(state)
    assert np.isposinf(actual) and np.isneginf(model_value)
    assert np.isfinite(gradient).all() and np.isfinite(model_gradient).all()
    np.testing.assert_allclose(gradient, -model_gradient, rtol=3e-10, atol=3e-9)


@pytest.mark.parametrize("formulation", ["joint", "marginal"])
def test_executed_model_and_objective_gradients_match_finite_difference(case, formulation):
    data, model, theta, _ = case
    state = jnp.r_[theta, jnp.array([0.15, -0.4])] if formulation == "joint" else theta

    def objective(x):
        return study.negative_log_posterior(x, data, model, formulation)

    value, gradient = jax.jit(jax.value_and_grad(objective))(state)
    model_gradient = jax.grad(lambda x: -model_log_prob(x, data, model, formulation))(state)
    assert np.isfinite(value) and np.isfinite(gradient).all()
    np.testing.assert_allclose(gradient, model_gradient, rtol=3e-10, atol=3e-9)
    direction = np.linspace(0.2, 0.8, len(state))
    direction /= np.linalg.norm(direction)
    step = 1e-6
    difference = (objective(state + step * direction) - objective(state - step * direction)) / (
        2 * step
    )
    np.testing.assert_allclose(gradient @ direction, difference, rtol=2e-5, atol=3e-6)


def test_flux_summary_uses_total_variance_and_conditional_mean_mcse():
    rng = np.random.default_rng(48)
    lens = rng.normal(size=(2, 1200, 4))
    means = 0.3 * lens[..., :2] + np.array([0.8, 0.2])
    covariance = np.broadcast_to(np.array([[0.04, -0.01], [-0.01, 0.02]]), (2, 1200, 2, 2))
    flux = means + rng.normal(size=means.shape) @ np.linalg.cholesky(covariance[0, 0]).T
    joint = np.concatenate((lens, flux), axis=-1)
    actual = study.compare_posteriors(joint, lens, means, covariance)
    expected_mean = study.mean_statistics(means)
    for i, name in enumerate(("f_s", "f_b")):
        np.testing.assert_allclose(actual[name]["marginal_mean"], means[..., i].mean())
        np.testing.assert_allclose(actual[name]["marginal_mean_mcse"], expected_mean["mcse"][i])
        np.testing.assert_allclose(
            actual[name]["marginal_sd"] ** 2,
            means[..., i].var() + covariance[0, 0, i, i],
        )


def test_fixed_settings_and_signed_flux_coordinates():
    settings = study.SETTINGS
    assert settings["data_seed"] == 44
    assert settings["sampler_seeds"] == {"joint": 18, "marginal": 19}
    assert settings["warmup"] == 1000 and settings["target_accept_prob"] == 0.99
    assert settings["target_lens_ess"] == 1000 and settings["maximum_draws_per_chain"] == 8000
    state = np.r_[study.experiment.PRIOR_LOC, [0.0, -10.0]]
    actual = study.physical_states(state, "joint")
    assert actual.shape == (6,) and actual[-1] < 0
    np.testing.assert_allclose(actual[-2:], study.flux_from_white(state[4:]))


def test_unreliable_early_autocorrelation_does_not_count_as_convergence(monkeypatch):
    def unreliable(values):
        raise RuntimeError("Short chain")

    monkeypatch.setattr(study.shared, "autocorrelation", unreliable)
    rng = np.random.default_rng(49)
    states = rng.normal(size=(2, 100, 4))
    fields = dict(diverging=np.zeros((2, 100), bool), num_steps=np.full((2, 100), 15))
    result = study.chain_diagnostics(states, fields, "marginal")
    assert not result["passed"] and not result["fatal"]
    assert result["minimum_lens_ess"] == 0
    fields["diverging"][0, 0] = True
    assert study.chain_diagnostics(states, fields, "marginal")["fatal"]
