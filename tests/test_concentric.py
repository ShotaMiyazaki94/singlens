"""Independent accuracy and differentiation checks for disk superposition."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from singlens.lensing.concentric import ConcentricDisk, UniformDiskQuadrature
from singlens.lensing.gould import GouldYoo
from singlens.lensing.point import pspl


@lru_cache(None)
def direct_integral(u, rho, a1=0.5):
    """Integrate the actual intensity in lens-centered polar coordinates.

    This reference never superposes disks. The radial area element cancels
    the point-lens pole, and scaling all distances by rho keeps the absolute
    integration tolerance useful for small sources.
    """
    ratio = abs(u) / rho

    def angular(theta):
        center = ratio * np.cos(theta)
        transverse = ratio * np.sin(theta)
        delta = np.sqrt(max(0.0, 1 - transverse**2))
        lo, hi = max(0.0, center - delta), center + delta

        def radial(x):
            mu = np.sqrt(max(0.0, 1 - (x - center) ** 2 - transverse**2))
            s = rho * x
            return (1 - a1 + a1 * mu) * (s * s + 2) / np.hypot(s, 2)

        return quad(radial, lo, hi, epsabs=2e-12, epsrel=2e-10, limit=150)[0]

    upper = np.pi if ratio < 1 else np.arcsin(min(1.0, 1 / ratio))
    value = quad(angular, 0, upper, epsabs=2e-12, epsrel=2e-10, limit=150)[0]
    return 2 * value / (np.pi * rho * (1 - a1 / 3))


@pytest.fixture(scope="module")
def concentric():
    return ConcentricDisk(n_disk=128, n_limb=64)


@pytest.mark.parametrize("rho", [1e-6, 0.01, 1.0, 10.0])
def test_uniform_center_and_contact_have_analytic_values(rho):
    source = UniformDiskQuadrature(128)
    expected = [
        np.hypot(rho, 2) / rho,
        2 * (rho + (1 + rho**2) * np.arctan(rho)) / (np.pi * rho**2),
    ]
    np.testing.assert_allclose(source.A(np.array([0.0, rho]), rho), expected, rtol=2e-10)
    derivative = jax.grad(lambda r: source.A(0.0, r).sum())(rho)
    np.testing.assert_allclose(derivative, -4 / (rho**2 * np.hypot(rho, 2)), rtol=1e-10)
    assert jax.grad(lambda u: source.A(u, rho).sum())(0.0) == 0.0


@pytest.mark.parametrize("rho", [1e-5, 0.1, 1.0, 10.0])
def test_concentric_against_direct_intensity_integral(concentric, rho):
    ratios = np.array([0.0, 0.3, 0.9, 0.999, 1.0, 1.001, 1.5, 10.0])
    expected = [direct_integral(float(z * rho), rho) for z in ratios]
    np.testing.assert_allclose(concentric.A(ratios * rho, rho), expected, rtol=2e-9)


def test_basis_is_flux_normalized_and_coefficient_is_dynamic(concentric):
    u, rho = jnp.array([0.0, 0.04, 0.11, 0.3]), 0.1
    disk, mu = concentric.basis(u, rho)
    np.testing.assert_allclose(disk, concentric.A(u, rho, a1=0.0), rtol=1e-13)
    np.testing.assert_allclose(mu, concentric.A(u, rho, a1=1.0), rtol=1e-13)
    for coefficient in (0.0, 0.2, 0.7, 1.0):
        expected = [direct_integral(float(x), rho, coefficient) for x in u]
        np.testing.assert_allclose(concentric.A(u, rho, a1=coefficient), expected, rtol=2e-9)
    coefficient = 0.37
    derivative = jax.jit(jax.jacfwd(lambda a: concentric.A(u, rho, a1=a)))(coefficient)
    expected = 6 * (mu - disk) / (3 - coefficient) ** 2
    np.testing.assert_allclose(derivative, expected, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("ratio", [0.5, 0.999, 1.001, 1.5])
def test_gradients_against_direct_intensity_integral(concentric, ratio):
    # The nonzero edge intensity gives an infinite slope at exact contact.
    # On either side the moving disk partition must differentiate correctly.
    rho, a1 = 0.1, 0.43
    u, step = ratio * rho, 1e-6 * rho
    theta = jnp.array([u, rho, a1])

    def evaluate(p):
        return concentric.A(p[0], p[1], a1=p[2]).sum()

    actual = jax.jit(jax.grad(evaluate))(theta)
    expected = [
        (direct_integral(u + step, rho, a1) - direct_integral(u - step, rho, a1)) / (2 * step),
        (direct_integral(u, rho + step, a1) - direct_integral(u, rho - step, a1)) / (2 * step),
        (direct_integral(u, rho, a1 + 1e-4) - direct_integral(u, rho, a1 - 1e-4)) / 2e-4,
    ]
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(theta), actual, rtol=2e-11, atol=1e-9)


@pytest.mark.parametrize("source", [UniformDiskQuadrature(32), ConcentricDisk(32, 16)])
def test_point_limit_invalid_radii_shapes_and_batching(source):
    u = jnp.array([[0.03, 0.08], [0.15, 0.8]])
    evaluate = jax.jit(source.A)
    np.testing.assert_allclose(evaluate(u, 0.0), pspl(u), rtol=1e-13)
    assert source.A(0.1, 0.02).shape == (1,)
    assert evaluate(u, 0.02).shape == u.shape
    np.testing.assert_allclose(evaluate(-u, 0.02), evaluate(u, 0.02), rtol=1e-13)
    for radius in (-0.1, np.nan, np.inf):
        assert np.isnan(evaluate(u, radius)).all()
    with pytest.raises(ValueError, match="scalar"):
        source.A(u, jnp.array([0.01, 0.02]))
    radii = jnp.array([0.0, 0.01, 0.1])
    batch = jax.jit(jax.vmap(source.A, in_axes=(None, 0)))(u, radii)
    np.testing.assert_allclose(batch, jnp.stack([source.A(u, r) for r in radii]), rtol=1e-12)
    np.testing.assert_allclose(
        jax.vmap(lambda x: source.A(x, 0.1)[0])(u.ravel()), source.A(u, 0.1).ravel()
    )


def test_far_point_is_independent_of_other_batch_members(concentric):
    # Inference must not change with the grouping or order of observation times.
    far = jnp.array([0.5, 1.0, 10.0])
    together = jnp.r_[0.01, far]

    def curve(p, positions):
        return concentric.A(positions, p[0], a1=p[1])

    theta = jnp.array([0.03, 0.4])
    np.testing.assert_allclose(curve(theta, far), curve(theta, together)[1:], rtol=1e-13)
    derivatives = jax.jit(jax.jacfwd(curve))
    np.testing.assert_allclose(
        derivatives(theta, far), derivatives(theta, together)[1:], rtol=1e-11, atol=1e-12
    )
    expected = [direct_integral(float(u), 0.03, 0.4) for u in far]
    np.testing.assert_allclose(curve(theta, far), expected, rtol=2e-9)


def test_quadrature_refinement_improves_independent_error():
    ratios, rho = np.array([0.3, 0.9, 0.999, 1.0, 1.001, 1.5]), 0.1
    expected = np.array([direct_integral(float(z * rho), rho) for z in ratios])
    errors = []
    for n_disk, n_limb in ((16, 8), (32, 16), (64, 32)):
        source = ConcentricDisk(n_disk, n_limb)
        errors.append(np.max(abs(np.asarray(source.A(ratios * rho, rho)) / expected - 1)))
    assert errors[1] < errors[0] / 10
    assert errors[2] < errors[1] / 10


@pytest.mark.parametrize("source", [ConcentricDisk(16, 8), GouldYoo(n_disk=16, n_limb=8)])
def test_zero_radius_alignment_and_coefficient_domain(source):
    evaluate = jax.jit(lambda a: source.A(0.0, 0.0, a1=a))
    for coefficient in (0.0, 0.5, 1.0):
        assert np.isposinf(evaluate(coefficient)).all()
        assert jax.grad(lambda a: evaluate(a).sum())(coefficient) == 0
    for coefficient in (-0.01, 1.01, np.nan, np.inf):
        assert np.isnan(source.A(0.05, 0.1, a1=coefficient)).all()


@pytest.mark.parametrize("kwargs", [{"n_disk": 0}, {"n_limb": 0}, {"n_disk": 3.5}])
def test_invalid_quadrature_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        ConcentricDisk(**kwargs)


@pytest.mark.parametrize("order", [0, 2, 4])
def test_gould_uniform_center_moments(order):
    """Analytic source moments verify coefficients of the corrected expansion."""
    source = GouldYoo(n_disk=96, n_limb=48, order=order, a1=0.0)
    rho = 0.17
    expected = 2 / rho
    slope = -2 / rho**2
    if order >= 2:
        expected += rho / 4
        slope += 1 / 4
    if order >= 4:
        expected -= rho**3 / 64
        slope -= 3 * rho**2 / 64
    np.testing.assert_allclose(source.A(0.0, rho), expected, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(lambda r: source.A(0.0, r).sum())(rho), slope, rtol=1e-12)


def test_gould_corrections_reduce_actual_small_source_error():
    ratios, rho = np.array([0.0, 0.35, 0.7, 1.2]), 0.2
    reference = np.array([direct_integral(float(z * rho), rho) for z in ratios])
    errors = []
    for order in (0, 2, 4):
        source = GouldYoo(n_disk=128, n_limb=64, order=order)
        values = np.asarray(source.A(ratios * rho, rho))
        errors.append(np.max(np.abs(values / reference - 1)))
    assert errors[1] < 0.2 * errors[0]
    assert errors[2] < 0.2 * errors[1]


def test_corrected_gould_derivatives_and_domain():
    source = GouldYoo(n_disk=128, n_limb=64, order=4)
    theta = jnp.array([0.06, 0.1, 0.4])

    def evaluate(p):
        return source.A(p[0], p[1], a1=p[2]).sum()

    actual = jax.jit(jax.grad(evaluate))(theta)
    expected = []
    for i, step in enumerate((1e-6, 1e-6, 1e-4)):
        plus, minus = np.asarray(theta).copy(), np.asarray(theta).copy()
        plus[i] += step
        minus[i] -= step
        expected.append((direct_integral(*plus) - direct_integral(*minus)) / (2 * step))
    np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=1e-5)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(theta), actual, rtol=1e-10)
    assert np.isnan(source.A(1.1, 1.0)).all()
    assert np.isnan(source.A(1.0, 1.0)).all()
    np.testing.assert_allclose(source.A(jnp.array([0.2, 3.0]), 0.0), pspl(jnp.array([0.2, 3.0])))


@pytest.mark.parametrize("order", [-1, 1, 3, 6])
def test_gould_rejects_unsupported_order(order):
    with pytest.raises(ValueError):
        GouldYoo(order=order)
