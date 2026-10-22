"""Astrometric image moments, including independent source integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import singlens as sl
from tools.validation.astrometry_accuracy import lens_centered_shift, reference_shift, wing_shift


@pytest.fixture(scope="module", params=[("uniform", None), ("linear", 0.6), ("mu2", None)])
def astrometric_model(request):
    source, a1 = request.param
    return sl.Model(source=source), source, a1


@pytest.mark.parametrize("rho", [1e-6, 0.01, 0.1, 1.0])
def test_astrometry_against_independent_image_moment(astrometric_model, rho):
    model, source, a1 = astrometric_model
    u = rho * np.array([0, 0.001, 0.1, 0.5, 0.99, 1, 1.01, 2, 5, 20])
    expected = [reference_shift(float(x), rho, source, a1) for x in u]
    actual = jax.jit(model.astrometric_shift_at)(u, rho=rho, a1=a1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3 * min(rho, 1))


@pytest.mark.parametrize("source,a1", [("uniform", None), ("linear", 0.6), ("mu2", None)])
def test_independent_reference_overlap(source, a1):
    for rho in (1e-5, 0.1, 10.0):
        u = 3 * rho
        expected = lens_centered_shift(u, rho, source, a1)
        np.testing.assert_allclose(wing_shift(u, rho, source, a1), expected, rtol=2e-9, atol=1e-12)
        np.testing.assert_allclose(
            wing_shift(u, rho, source, a1, order=96), expected, rtol=2e-9, atol=1e-12
        )


@pytest.mark.parametrize("rho", [1e-6, 0.01, 1.0, 10.0])
def test_uniform_center_slope_and_limb_zero(rho):
    model = sl.Model(source="uniform")

    def shift(x):
        return model.astrometric_shift_at(x, rho=rho).real

    assert shift(0.0) == 0
    np.testing.assert_allclose(jax.grad(shift)(0.0), -1 / (rho * rho + 4), rtol=1e-10)
    np.testing.assert_allclose(shift(rho * 1e-5) / (rho * 1e-5), -1 / (rho * rho + 4), rtol=1e-7)
    np.testing.assert_allclose(shift(rho), 0.0, atol=1e-3 * min(rho, 1))
    np.testing.assert_allclose(lens_centered_shift(rho, rho), 0.0, atol=2e-12 * rho)
    assert shift(rho * 0.5) < 0
    assert shift(rho * 2) > 0


def test_center_direction_jacobian_is_finite_and_isotropic(astrometric_model):
    model, source, a1 = astrometric_model
    rho = 0.1

    def vector_shift(xy):
        value = model.astrometric_shift_at(xy[0] + 1j * xy[1], rho=rho, a1=a1)
        return jnp.array([value.real, value.imag])

    forward = jax.jit(jax.jacfwd(vector_shift))(jnp.zeros(2))
    reverse = jax.jacrev(vector_shift)(jnp.zeros(2))
    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, reverse, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(forward, jnp.eye(2) * forward[0, 0], atol=1e-12)
    h = rho * 1e-5
    fd = (vector_shift(jnp.array([h, 0])) - vector_shift(jnp.array([-h, 0]))) / (2 * h)
    np.testing.assert_allclose(forward[:, 0], fd, rtol=1e-7, atol=1e-10)
    reference_step = rho * 1e-4
    expected_slope = reference_shift(reference_step, rho, source, a1) / reference_step
    np.testing.assert_allclose(forward[0, 0], expected_slope, rtol=1e-6, atol=1e-10)


def test_point_limit_rotation_and_shapes(astrometric_model):
    model, _, a1 = astrometric_model
    position = jnp.array([[0, 0.1 + 0.2j], [-0.3 + 0.1j, 30 - 40j]])
    expected = position / (jnp.abs(position) ** 2 + 2)
    np.testing.assert_allclose(
        model.astrometric_shift_at(position, rho=0.0, a1=a1), expected, rtol=1e-12, atol=1e-15
    )
    rho, rotation = 0.1, np.exp(0.73j)
    original = model.astrometric_shift_at(position, rho=rho, a1=a1)
    rotated = model.astrometric_shift_at(position * rotation, rho=rho, a1=a1)
    np.testing.assert_allclose(rotated, rotation * original, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        model.astrometric_shift_at(-position, rho=rho, a1=a1), -original, rtol=1e-12, atol=1e-12
    )
    assert original.shape == position.shape
    assert model.astrometric_shift_at(0.2 + 0.1j, rho=rho, a1=a1).shape == ()


def test_far_field_and_batch_independence(astrometric_model):
    model, source, a1 = astrometric_model
    evaluate = jax.jit(model.astrometric_shift_at)
    for rho in (1e-6, 0.01, 1.0, 10.0):
        u = np.maximum(np.array([20, 30, 100]) * rho, [0.01, 1, 100])
        expected = [wing_shift(float(x), rho, source, a1) for x in u]
        all_far = evaluate(u, rho=rho, a1=a1)
        mixed = evaluate(jnp.r_[rho * 0.5, u], rho=rho, a1=a1)[1:]
        np.testing.assert_allclose(all_far, mixed, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(all_far, expected, rtol=2e-5, atol=1e-10)


def test_position_and_radius_gradients_against_independent_integral(astrometric_model):
    _, source, a1 = astrometric_model
    # Resolving derivatives needs a finer grid than the default value-accuracy
    # target. The independent comparison still includes the far-field expansion.
    model = sl.Model(source=source, fft=sl.FFTConfig(n_fft=8192))
    rho, step = 0.1, 1e-5
    derivative = jax.jit(
        jax.grad(lambda u, r: model.astrometric_shift_at(u, rho=r, a1=a1).real, argnums=(0, 1))
    )
    for ratio in (0.5, 1.5, 25.0):
        u = ratio * rho
        du = (
            reference_shift(u + step, rho, source, a1) - reference_shift(u - step, rho, source, a1)
        ) / (2 * step)
        dr = (
            reference_shift(u, rho + step, source, a1) - reference_shift(u, rho - step, source, a1)
        ) / (2 * step)
        np.testing.assert_allclose(derivative(u, rho), [du, dr], rtol=1e-2, atol=1e-5)


def test_default_autodiff_matches_its_finite_difference(astrometric_model):
    model, _, a1 = astrometric_model

    def value(parameters):
        return model.astrometric_shift_at(parameters[0], rho=parameters[1], a1=a1).real

    evaluate = jax.jit(value)
    derivative = jax.jit(jax.grad(value))
    for rho in (1e-6, 0.1):
        parameters = jnp.array([0.5 * rho, rho])
        step = rho * 1e-5
        displacements = jnp.eye(2) * step
        finite_difference = jnp.array(
            [
                (evaluate(parameters + displacement) - evaluate(parameters - displacement))
                / (2 * step)
                for displacement in displacements
            ]
        )
        np.testing.assert_allclose(derivative(parameters), finite_difference, rtol=2e-5, atol=1e-7)


def test_dynamic_limb_coefficient_gradient():
    model = sl.Model(source="linear")
    rho, a1, u, step = 0.1, 0.4, 0.05, 1e-4
    evaluate = jax.jit(lambda a: model.astrometric_shift_at(u, rho=rho, a1=a).real)
    fd = (
        reference_shift(u, rho, "linear", a1 + step) - reference_shift(u, rho, "linear", a1 - step)
    ) / (2 * step)
    np.testing.assert_allclose(jax.grad(evaluate)(a1), fd, rtol=1e-2, atol=1e-6)
    assert evaluate(0.0) != evaluate(1.0)


def test_small_source_boundary_and_vmap(astrometric_model):
    model, source, a1 = astrometric_model
    radii = jnp.array([1e-6, 9.99e-5, 1.001e-4])

    def evaluate(rho):
        return model.astrometric_shift_at(0.5 * rho, rho=rho, a1=a1)

    actual = jax.jit(jax.vmap(evaluate))(radii)
    expected = [reference_shift(0.5 * float(r), float(r), source, a1) for r in radii]
    np.testing.assert_allclose(actual / radii, np.asarray(expected) / radii, rtol=0, atol=1e-3)
    assert np.isfinite(jax.jit(jax.vmap(jax.grad(lambda r: evaluate(r).real)))(radii)).all()
