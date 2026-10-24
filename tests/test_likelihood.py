import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens.photometry import (
    FluxMarginalization,
    fit_flux,
    marginalize_flux,
    nll_ulens,
    nll_ulens_general,
)


@pytest.mark.parametrize("err", [1.0, 0.2, np.array([0.1, 0.2, 0.3, 0.4])])
def test_linear_fit_against_numpy(err):
    x = np.array([1.0, 2.0, 3.0, 6.0])
    y = np.array([2.0, 2.9, 4.2, 7.1])
    sigma = np.broadcast_to(err, x.shape)
    design = np.column_stack([x, np.ones_like(x)])
    beta = np.linalg.lstsq(design / sigma[:, None], y / sigma, rcond=None)[0]
    cov = np.linalg.inv((design / sigma[:, None]).T @ (design / sigma[:, None]))
    result = jax.jit(fit_flux)(x, y, err)
    np.testing.assert_allclose([result.f_s, result.f_b], beta)
    np.testing.assert_allclose([result.f_s_err, result.f_b_err], np.sqrt(np.diag(cov)))
    np.testing.assert_allclose(result.chi2, np.sum(((y - design @ beta) / sigma) ** 2))


def test_marginal_likelihood_against_dense_gaussian():
    flux = np.array([2.0, 2.3, 1.4, 0.8])
    M = np.column_stack([[1.2, 1.5, 1.1, 1.0], np.ones(4)])
    variance = np.array([0.3, 0.2, 0.1, 0.2])
    prior = np.diag([2.0, 1.0])
    C = np.diag(variance)
    total = C + M @ prior @ M.T
    expected = 0.5 * (flux @ np.linalg.solve(total, flux) + np.linalg.slogdet(total)[1])
    np.testing.assert_allclose(nll_ulens(flux, M, variance, 2.0, 1.0), expected, rtol=1e-12)
    np.testing.assert_allclose(
        nll_ulens_general(flux, M, C, np.zeros(2), prior), expected, rtol=1e-12
    )
    mean = np.array([1.0, 0.2])
    prior[0, 1] = prior[1, 0] = 0.2
    C[0, 1] = C[1, 0] = 0.03
    r = flux - M @ mean
    total = C + M @ prior @ M.T
    expected = 0.5 * (r @ np.linalg.solve(total, r) + np.linalg.slogdet(total)[1])
    np.testing.assert_allclose(nll_ulens_general(flux, M, C, mean, prior), expected, rtol=1e-12)
    assert np.isfinite(
        jax.grad(lambda f: nll_ulens(f, M, variance, 2.0, 1.0))(jnp.array(flux))
    ).all()
    np.testing.assert_allclose(
        jax.jit(jax.grad(nll_ulens_general))(flux, M, C, mean, prior),
        np.linalg.solve(total, r),
        rtol=1e-12,
    )


@pytest.mark.parametrize("magnification", [[1.3], [1.0, 1.0, 1.0], [1.1, 2.5, 1.3]])
@pytest.mark.parametrize("error", [0.15, np.array([0.1, 0.2, 0.3])])
def test_flux_marginalization_matches_dense_gaussian(magnification, error):
    a = np.array(magnification)
    error = np.broadcast_to(error, (3,))[: a.size] if np.ndim(error) else error
    flux = np.arange(a.size) * 0.2 + 0.9 * a + 0.1
    mean, prior = np.array([0.8, 0.2]), np.array([[0.3, -0.12], [-0.12, 0.2]])
    design = np.column_stack((a, np.ones_like(a)))
    noise = np.diag(np.broadcast_to(error, a.shape) ** 2)
    total = noise + design @ prior @ design.T
    residual = flux - design @ mean
    expected_logp = -0.5 * (
        residual @ np.linalg.solve(total, residual)
        + np.linalg.slogdet(total)[1]
        + a.size * np.log(2 * np.pi)
    )
    expected_mean = mean + prior @ design.T @ np.linalg.solve(total, residual)
    expected_cov = prior - prior @ design.T @ np.linalg.solve(total, design @ prior)
    result = jax.jit(marginalize_flux)(a, flux, error, prior_mean=mean, prior_covariance=prior)
    assert isinstance(result, FluxMarginalization)
    np.testing.assert_allclose(result.log_likelihood, expected_logp, rtol=1e-12)
    np.testing.assert_allclose(result.mean, expected_mean, rtol=1e-12)
    np.testing.assert_allclose(result.covariance, expected_cov, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        nll_ulens_general(flux, design, noise, mean, prior),
        -result.log_likelihood - a.size / 2 * np.log(2 * np.pi),
        rtol=1e-12,
    )


def test_flux_marginalization_gradient_against_dense_gaussian():
    a = jnp.array([1.01, 1.8, 2.6, 1.1])
    flux = jnp.array([1.0, 1.55, 2.25, 1.2])
    error = jnp.array([0.1, 0.15, 0.2, 0.12])
    mean, prior = jnp.array([0.8, 0.2]), jnp.array([[0.2, -0.12], [-0.12, 0.1]])

    def actual(a):
        return marginalize_flux(
            a, flux, error, prior_mean=mean, prior_covariance=prior
        ).log_likelihood

    def dense(a):
        design = jnp.column_stack((a, jnp.ones_like(a)))
        total = jnp.diag(error**2) + design @ prior @ design.T
        residual = flux - design @ mean
        return -0.5 * (
            residual @ jnp.linalg.solve(total, residual)
            + jnp.linalg.slogdet(total)[1]
            + a.size * jnp.log(2 * jnp.pi)
        )

    np.testing.assert_allclose(jax.jit(jax.grad(actual))(a), jax.grad(dense)(a), rtol=1e-11)
    # The conditional moments remain differentiable for posterior reconstruction.
    moments = jax.jacfwd(
        lambda a: marginalize_flux(a, flux, error, prior_mean=mean, prior_covariance=prior)
    )(a)
    assert all(np.isfinite(value).all() for value in moments)


def test_nearly_constant_magnification_with_large_flux_baseline():
    a = np.ones(12) + np.linspace(-1e-10, 1e-10, 12)
    mean, prior = np.array([0.8, 1e8]), np.array([[0.5, -0.2], [-0.2, 0.3]])
    flux = a * mean[0] + mean[1] + np.linspace(-0.01, 0.01, a.size)
    compiled = jax.jit(marginalize_flux)
    result = compiled(a, flux, 0.02, prior_mean=mean, prior_covariance=prior)
    # A baseline translation changes only the conditional blend mean.
    centered = compiled(
        a, flux - mean[1], 0.02, prior_mean=np.array([mean[0], 0.0]), prior_covariance=prior
    )
    np.testing.assert_allclose(result.log_likelihood, centered.log_likelihood, atol=1e-6)
    np.testing.assert_allclose(result.mean - np.array([0.0, mean[1]]), centered.mean, atol=1e-8)
    np.testing.assert_allclose(result.covariance, centered.covariance, rtol=1e-12)
    assert np.linalg.eigvalsh(result.covariance).min() > 0


@pytest.mark.parametrize("magnification", [1.0, 1000.0])
@pytest.mark.parametrize("prior_variance", [1e10, 1e12])
def test_diffuse_prior_constant_curve_retains_flux_uncertainty(magnification, prior_variance):
    # A constant curve measures only a*f_s + f_b. Even a very broad proper
    # prior must retain its uncertainty in the perpendicular flux direction.
    count, error = 10, 0.01
    a = jnp.full(count, magnification)
    prior_mean = jnp.array([1.0, 0.0])
    prior = prior_variance * jnp.eye(2)
    information = count / error**2
    denominator = 1 + prior_variance * information * (magnification**2 + 1)
    expected_covariance = (
        prior_variance
        / denominator
        * np.array(
            [
                [1 + prior_variance * information, -prior_variance * information * magnification],
                [
                    -prior_variance * information * magnification,
                    1 + prior_variance * information * magnification**2,
                ],
            ]
        )
    )

    def evaluate(a, flux):
        return marginalize_flux(a, flux, error, prior_mean=prior_mean, prior_covariance=prior)

    for offset in (0.0, 0.03):
        flux = jnp.full(count, magnification + offset)
        result = jax.jit(evaluate)(a, flux)
        residual = float(flux[0]) - magnification
        expected_logp = -0.5 * (
            count * np.log(2 * np.pi * error**2)
            + np.log(denominator)
            + information * residual**2 / denominator
        )
        expected_mean = np.array(prior_mean) + (
            prior_variance * information * residual / denominator
        ) * np.array([magnification, 1.0])
        np.testing.assert_allclose(result.log_likelihood, expected_logp, rtol=1e-13)
        np.testing.assert_allclose(result.mean, expected_mean, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(result.covariance, expected_covariance, rtol=1e-13)

    derivative = jax.jit(jax.grad(lambda a: evaluate(a, a * 0 + magnification).log_likelihood))(a)
    expected_derivative = -prior_variance * information * magnification / denominator / count
    np.testing.assert_allclose(derivative, expected_derivative, rtol=1e-13)


@pytest.mark.parametrize(
    "design",
    [np.zeros((3, 2)), [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]],
)
def test_marginal_likelihood_handles_zero_design_columns(design):
    design = jnp.array(design)
    flux, noise = jnp.array([0.2, 0.9, -0.1]), jnp.diag(jnp.array([0.1, 0.2, 0.3]))
    mean, prior = jnp.array([0.4, -0.2]), jnp.array([[0.3, 0.1], [0.1, 0.2]])

    def dense(design):
        total = noise + design @ prior @ design.T
        residual = flux - design @ mean
        return 0.5 * (residual @ jnp.linalg.solve(total, residual) + jnp.linalg.slogdet(total)[1])

    def actual(design):
        return nll_ulens_general(flux, design, noise, mean, prior)

    np.testing.assert_allclose(jax.jit(actual)(design), dense(design), rtol=1e-12)
    np.testing.assert_allclose(
        jax.jit(jax.grad(actual))(design), jax.grad(dense)(design), rtol=1e-12, atol=1e-14
    )


def test_flux_marginalization_vmap_and_integer_input():
    batched = jax.jit(
        jax.vmap(
            lambda f: marginalize_flux(
                [1, 2, 3], f, 1, prior_mean=[1, 0], prior_covariance=[[1, 0], [0, 1]]
            )
        )
    )(jnp.array([[1, 2, 3], [2, 3, 4]]))
    assert batched.log_likelihood.shape == (2,)
    assert batched.mean.shape == (2, 2)
    assert batched.covariance.shape == (2, 2, 2)
    assert all(np.isfinite(value).all() for value in batched)


@pytest.mark.parametrize(
    "field,value",
    [
        ("magnification", [1.0, np.nan]),
        ("observed_flux", [1.0, np.inf]),
        ("flux_err", 0.0),
        ("flux_err", -0.1),
        ("flux_err", np.nan),
        ("flux_err", np.inf),
        ("prior_mean", [0.0, np.nan]),
        ("prior_covariance", [[1.0, 1.0], [1.0, 1.0]]),
        ("prior_covariance", [[1.0, 0.0], [0.0, -1.0]]),
        ("prior_covariance", [[1.0, 0.1], [0.2, 1.0]]),
        ("prior_covariance", [[1.0, np.nan], [np.nan, 1.0]]),
    ],
)
def test_flux_marginalization_invalid_numeric_inputs(field, value):
    arguments = dict(
        magnification=[1.0, 2.0],
        observed_flux=[1.0, 1.5],
        flux_err=0.1,
        prior_mean=[0.8, 0.2],
        prior_covariance=[[0.2, 0.0], [0.0, 0.1]],
    )
    arguments[field] = value
    for function in (marginalize_flux, jax.jit(marginalize_flux)):
        assert all(np.isnan(value).all() for value in function(**arguments))


@pytest.mark.parametrize(
    "field,value",
    [
        ("magnification", [[1.0, 2.0]]),
        ("observed_flux", [1.0]),
        ("flux_err", [0.1]),
        ("prior_mean", [0.8]),
        ("prior_covariance", [0.2, 0.1]),
        ("prior_mean", [0.8 + 1j, 0.2]),
    ],
)
def test_flux_marginalization_invalid_shapes(field, value):
    arguments = dict(
        magnification=[1.0, 2.0],
        observed_flux=[1.0, 1.5],
        flux_err=0.1,
        prior_mean=[0.8, 0.2],
        prior_covariance=[[0.2, 0.0], [0.0, 0.1]],
    )
    arguments[field] = value
    with pytest.raises(ValueError):
        marginalize_flux(**arguments)
    with pytest.raises(ValueError, match="nonempty"):
        marginalize_flux([], [], 1.0, prior_mean=[0, 0], prior_covariance=np.eye(2))


@pytest.mark.parametrize("bad", [np.diag([0.1, -0.2]), [[0.1, 0.0], [0.01, 0.2]]])
def test_general_marginalization_rejects_invalid_noise_covariance(bad):
    assert np.isnan(
        jax.jit(nll_ulens_general)([1.0, 2.0], [[1.0, 1.0], [2.0, 1.0]], bad, [0.0, 0.0], np.eye(2))
    )
