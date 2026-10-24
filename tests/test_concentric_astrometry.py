"""Flux-weighted disk moments checked against independent image integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens.lensing.concentric import ConcentricDisk, UniformDiskQuadrature
from singlens.lensing.point import point_source_shift, pspl
from tools.validation.astrometry_accuracy import reference_shift, wing_shift


@pytest.fixture(scope="module")
def solver():
    return ConcentricDisk(n_disk=96, n_limb=64)


@pytest.mark.parametrize("rho", [1e-6, 0.1, 1.0, 10.0])
@pytest.mark.parametrize("a1", [0.0, 0.6, 1.0])
def test_joint_shift_against_independent_image_integration(solver, rho, a1):
    ratios = np.array([0.0, 0.001, 0.2, 0.7, 0.999, 1.0, 1.001, 1.7, 5.0])
    expected = [reference_shift(float(z * rho), rho, "linear", a1) for z in ratios]
    flux, shift = jax.jit(solver.joint)(ratios * rho, rho, a1=a1)
    np.testing.assert_allclose(shift, expected, rtol=0, atol=2e-8 * min(rho, 1 / rho))
    np.testing.assert_allclose(flux, solver.A(ratios * rho, rho, a1=a1), rtol=2e-10)


@pytest.mark.parametrize("rho", [1e-6, 0.1, 1.0, 10.0])
def test_exact_uniform_center_slope_and_contact(solver, rho):
    def shift(u):
        return solver.shift(u, rho, a1=0.0)

    assert shift(0.0) == 0
    expected = -1 / (rho**2 + 4)
    np.testing.assert_allclose(jax.jit(jax.grad(shift))(0.0), expected, rtol=3e-12)
    np.testing.assert_allclose(shift(1e-5 * rho) / (1e-5 * rho), expected, rtol=2e-8)
    np.testing.assert_allclose(shift(rho), 0.0, atol=3e-11 * min(rho, 1 / rho))
    assert shift(0.5 * rho) < 0
    assert shift(2 * rho) > 0


def test_joint_basis_mixes_flux_weighted_moments(solver):
    rho = 0.3
    u = jnp.array([0.0, 0.001, 0.15, 0.3, 0.6, 3.0])
    f0, fm, c0, cm = solver.joint_basis(u, rho)
    np.testing.assert_allclose(f0, solver.basis(u, rho)[0], rtol=2e-12)
    np.testing.assert_allclose(fm, solver.basis(u, rho)[1], rtol=2e-12)
    np.testing.assert_allclose(c0[0], -1 / (rho**2 + 4), rtol=2e-12)
    assert np.isfinite(np.asarray([c0, cm])).all()
    for coefficient in (0.0, 0.37, 1.0):
        gamma = 2 * coefficient / (3 - coefficient)
        expected_flux = (1 - gamma) * f0 + gamma * fm
        expected_shift = u * ((1 - gamma) * f0 * c0 + gamma * fm * cm) / expected_flux
        flux, shift = solver.joint(u, rho, a1=coefficient)
        np.testing.assert_allclose(flux, expected_flux, rtol=2e-12)
        np.testing.assert_allclose(shift, expected_shift, rtol=2e-11, atol=2e-13)


@pytest.mark.parametrize("coefficient", [0.0, 0.6, 1.0])
def test_cartesian_origin_jacobian_and_rotation(solver, coefficient):
    rho = 0.1

    def cartesian(xy):
        value = solver.shift(xy[0] + 1j * xy[1], rho, a1=coefficient)
        return jnp.array([value.real, value.imag])

    origin = jnp.zeros(2)
    forward = jax.jit(jax.jacfwd(cartesian))(origin)
    reverse = jax.jacrev(cartesian)(origin)
    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, reverse, rtol=2e-11, atol=2e-13)
    h = rho * 1e-4
    slope = reference_shift(h, rho, "linear", coefficient) / h
    np.testing.assert_allclose(forward, np.eye(2) * slope, rtol=2e-7, atol=2e-11)
    positions = jnp.array([[0.0, -0.03, 0.02 + 0.07j], [0.1j, -0.2 + 0.4j, 3.0]])
    rotation = np.exp(0.73j)
    flux, shift = solver.joint(positions, rho, a1=coefficient)
    rotated_flux, rotated = solver.joint(positions * rotation, rho, a1=coefficient)
    np.testing.assert_allclose(rotated_flux, flux, rtol=2e-12)
    np.testing.assert_allclose(rotated, rotation * shift, rtol=2e-11, atol=2e-13)
    np.testing.assert_allclose(solver.shift(-positions, rho, a1=coefficient), -shift, atol=2e-13)
    assert flux.shape == shift.shape == positions.shape
    assert solver.joint(0.1, rho, a1=coefficient)[0].shape == ()
    assert solver.shift(0.1 + 0.2j, rho, a1=coefficient).shape == ()


@pytest.mark.parametrize("ratio", [0.5, 0.999, 1.001, 1.5, 30.0])
def test_joint_gradient_against_independent_moments(solver, ratio):
    rho, a1 = 0.1, 0.43
    theta = jnp.array([ratio * rho, rho, a1])

    def evaluate(p):
        return solver.shift(p[0], p[1], a1=p[2])

    expected = []
    for i, step in enumerate((1e-6 * rho, 1e-6 * rho, 1e-4)):
        plus, minus = np.array(theta), np.array(theta)
        plus[i] += step
        minus[i] -= step
        high = reference_shift(plus[0], plus[1], "linear", plus[2])
        low = reference_shift(minus[0], minus[1], "linear", minus[2])
        expected.append((high - low) / (2 * step))
    actual = jax.jit(jax.grad(evaluate))(theta)
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
    np.testing.assert_allclose(jax.jacfwd(evaluate)(theta), actual, rtol=2e-10, atol=2e-12)


def test_long_tail_and_batch_invariance(solver):
    rho, coefficient = 0.3, 0.6
    distant = jnp.array([3.0, 30.0, 300.0, 1e4, 1e8, 1e12])
    expected = [wing_shift(float(u), rho, "linear", coefficient) for u in distant]
    actual = solver.shift(distant, rho, a1=coefficient)
    np.testing.assert_allclose(actual, expected, rtol=3e-10, atol=1e-15)
    np.testing.assert_allclose(
        actual, solver.shift(jnp.r_[0.5 * rho, distant], rho, a1=coefficient)[1:], rtol=1e-13
    )

    def evaluate(p, positions):
        return solver.shift(positions, p[0], a1=p[1])

    parameters = jnp.array([rho, coefficient])
    derivative = jax.jit(jax.jacfwd(evaluate))
    np.testing.assert_allclose(
        derivative(parameters, distant),
        derivative(parameters, jnp.r_[0.5 * rho, distant])[1:],
        rtol=2e-11,
        atol=2e-13,
    )
    huge = jnp.array([1e200, -1e200, 1e200 + 1e200j])
    flux, shift = solver.joint(huge, rho, a1=coefficient)
    np.testing.assert_allclose(flux, 1.0, rtol=2e-14)
    np.testing.assert_allclose(shift, point_source_shift(huge), rtol=2e-14, atol=0)


def test_point_limits_invalid_parameters_and_vmap(solver):
    positions = jnp.array([[0, -0.01, 0.2], [0.1 + 0.4j, -3j, 100]])
    for coefficient in (0.0, 0.5, 1.0):
        flux, shift = jax.jit(solver.joint)(positions, 0.0, a1=coefficient)
        np.testing.assert_allclose(flux, pspl(positions), rtol=1e-14)
        np.testing.assert_allclose(shift, point_source_shift(positions), rtol=1e-14)
        assert jax.grad(lambda a: solver.shift(0.0, 0.0, a1=a))(coefficient) == 0
    np.testing.assert_allclose(jax.grad(lambda u: solver.shift(u, 0.0))(0.0), 0.5, rtol=1e-14)
    for rho in (-1.0, np.nan, np.inf):
        flux, shift = solver.joint(positions, rho)
        assert np.isnan(flux).all() and np.isnan(shift).all()
    for coefficient in (-0.1, 1.1, np.nan, np.inf):
        flux, shift = solver.joint(positions, 0.1, a1=coefficient)
        assert np.isnan(flux).all() and np.isnan(shift).all()
    with pytest.raises(ValueError, match="scalar"):
        solver.shift(0.1, jnp.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="real"):
        solver.shift(0.1, 0.1 + 0.01j)
    radii = jnp.array([0.0, 0.01, 0.1])
    batch = jax.jit(jax.vmap(solver.joint, in_axes=(None, 0)))(positions, radii)
    eager = [solver.joint(positions, rho) for rho in radii]
    for index in (0, 1):
        np.testing.assert_allclose(
            batch[index], jnp.stack([row[index] for row in eager]), rtol=2e-12
        )


def test_uniform_solver_exposes_consistent_joint_observables(solver):
    uniform = UniformDiskQuadrature(96)
    positions, rho = jnp.array([0.0, -0.01, 0.05 + 0.07j, 0.1, 3.0]), 0.1
    flux, shift = jax.jit(uniform.joint)(positions, rho)
    reference_flux, reference_shift = solver.joint(positions, rho, a1=0.0)
    np.testing.assert_allclose(flux, uniform.A(positions, rho), rtol=2e-12)
    np.testing.assert_allclose(flux, reference_flux, rtol=2e-12)
    np.testing.assert_allclose(shift, reference_shift, rtol=2e-12, atol=2e-14)
    np.testing.assert_allclose(
        jax.grad(lambda u: uniform.shift(u, rho))(0.0), -1 / (rho**2 + 4), rtol=2e-12
    )
    for actual, expected in zip(
        uniform.joint(positions, 0.0), (pspl(positions), point_source_shift(positions))
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-14)


@pytest.mark.parametrize("coefficient", [0.0, 0.6])
def test_joint_and_separate_preserve_extreme_wings(coefficient):
    solver = ConcentricDisk(n_disk=32, n_limb=16)
    position = jnp.array([[-1e200, 1e200], [1e200 + 1e200j, -1e200 + 1e200j]])
    for rho in (0.0, 0.1):
        flux, shift = jax.jit(solver.joint)(position, rho, a1=coefficient)
        np.testing.assert_allclose(flux, 1.0, rtol=2e-13)
        np.testing.assert_allclose(shift, point_source_shift(position), rtol=2e-13, atol=0)
        np.testing.assert_allclose(flux, solver.A(position, rho, a1=coefficient), rtol=2e-13)
        np.testing.assert_allclose(
            shift,
            solver.shift(position, rho, a1=coefficient),
            rtol=2e-13,
            atol=0,
        )
        assert flux.shape == shift.shape == position.shape
