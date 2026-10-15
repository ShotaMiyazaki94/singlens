import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens import fit_flux
from singlens.photometry import nll_ulens, nll_ulens_general


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
