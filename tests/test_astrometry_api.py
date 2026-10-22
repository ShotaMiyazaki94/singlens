"""Public astrometric interfaces, coordinate conventions, and JAX transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import singlens as sl
from singlens.lensing.point import point_source_shift
from singlens.lightcurve import _prepared_model


def _cartesian(value):
    return jnp.stack((jnp.real(value), jnp.imag(value)), axis=-1)


def test_point_shift_signed_positions_shapes_and_large_separations():
    model = sl.Model()
    for position in (0.0, -0.3, 0.3 + 0.4j, [[-2.0, 0.0], [0.2, 3.0]]):
        position = jnp.asarray(position)
        actual = jax.jit(model.astrometric_shift_at)(position)
        expected = position / (jnp.abs(position) ** 2 + 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-14)
        assert actual.shape == position.shape
        assert jnp.iscomplexobj(actual) == jnp.iscomplexobj(position)
    # Squaring these separations overflows, although the shifts are representable.
    positions = jnp.array([-1e200, 1e200, 1e200 + 1e200j])
    expected = jnp.array([-1e-200, 1e-200, (0.5 + 0.5j) * 1e-200])
    np.testing.assert_allclose(point_source_shift(positions), expected, rtol=1e-14, atol=0)


def test_point_alignment_has_finite_cartesian_derivatives():
    assert float(point_source_shift(0.0)) == 0
    np.testing.assert_allclose(jax.grad(point_source_shift)(0.0), 0.5, rtol=1e-14)

    def shift(xy):
        return _cartesian(point_source_shift(xy[0] + 1j * xy[1]))

    expected = jnp.eye(2) / 2
    np.testing.assert_allclose(jax.jit(jax.jacfwd(shift))(jnp.zeros(2)), expected, atol=1e-14)
    np.testing.assert_allclose(jax.jacrev(shift)(jnp.zeros(2)), expected, atol=1e-14)


def test_point_function_uses_signed_trajectory_and_preserves_time_shape():
    times = jnp.array([[-10.0, 0.0, 10.0]])
    position = times / 20 - 0.1j
    expected = position / (jnp.abs(position) ** 2 + 2)
    np.testing.assert_allclose(sl.astrometric_shift(times, 0.0, -0.1, 20.0), expected)
    assert sl.astrometric_shift(1.0, 0.0, 0.1, 20.0).shape == ()
    assert np.isnan(sl.astrometric_shift(times, 0.0, 0.1, -20.0)).all()


@pytest.mark.parametrize("source", ["uniform", "linear", "mu2"])
def test_finite_zero_radius_coordinate_covariance_and_shapes(source):
    model = sl.Model(source=source, fft=sl.FFTConfig(n_fft=128))
    profile = {"a1": 0.4} if source == "linear" else {}
    positions = jnp.array([[-0.02, 0.0, 0.03]])
    compiled = jax.jit(model.astrometric_shift_at)
    np.testing.assert_allclose(
        compiled(positions, rho=0.0, **profile), point_source_shift(positions), atol=1e-14
    )
    # Finite and zero-radius calls retain a representable far tail, including
    # integer coordinates whose squares would overflow before float promotion.
    for distant in (jnp.array([10**12, -(10**12)]), jnp.array([1e200, -1e200])):
        for rho in (0.0, 0.01):
            np.testing.assert_allclose(
                compiled(distant, rho=rho, **profile),
                point_source_shift(distant),
                rtol=1e-14,
                atol=0,
            )
    result = compiled(positions, rho=0.01, **profile)
    assert result.shape == positions.shape
    assert not jnp.iscomplexobj(result)
    rotation = jnp.exp(0.7j)
    np.testing.assert_allclose(
        compiled(positions * rotation, rho=0.01, **profile), rotation * result, atol=1e-13
    )
    assert compiled(0.03, rho=0.01, **profile).shape == ()
    assert float(result[0, 1]) == 0


@pytest.mark.parametrize("source", ["point", "uniform", "linear"])
def test_function_cold_jit_jacobians_and_event_batch(source):
    _prepared_model.cache_clear()
    config = None if source == "point" else sl.FFTConfig(n_fft=128)
    times = jnp.array([-0.41, 0.07, 0.33])
    theta = jnp.array([0.01, 0.006, 20.0, 0.012, 0.4])

    def evaluate(p):
        return sl.astrometric_shift(
            times,
            *p[:3],
            rho=None if source == "point" else p[3],
            a1=p[4] if source == "linear" else None,
            fft=config,
        )

    # Kernels must remain usable after a first call that occurs inside tracing.
    with jax.checking_leaks():
        actual = jax.jit(evaluate)(theta)
    np.testing.assert_allclose(evaluate(theta), actual, rtol=1e-12, atol=1e-14)

    def components(p):
        return _cartesian(evaluate(p))

    forward = jax.jit(jax.jacfwd(components))(theta)
    reverse = jax.jacrev(components)(theta)
    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, reverse, rtol=1e-9, atol=1e-10)
    for i in range(len(theta)):
        step = 1e-5 * max(abs(float(theta[i])), 0.001)
        offset = jnp.zeros_like(theta).at[i].set(step)
        difference = (components(theta + offset) - components(theta - offset)) / (2 * step)
        np.testing.assert_allclose(forward[..., i], difference, rtol=3e-4, atol=1e-6)
    batch = jnp.stack([theta, theta.at[0].set(-0.03)])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(batch), jnp.stack([evaluate(p) for p in batch]), atol=1e-14
    )
    model = sl.Model(source=source, fft=config)
    params = sl.Parameters(
        t0=theta[0],
        u0=theta[1],
        t_e=theta[2],
        rho=None if source == "point" else theta[3],
        a1=theta[4] if source == "linear" else None,
    )
    np.testing.assert_allclose(model.astrometric_shift(times, params), actual, rtol=1e-12)
    derivative = jax.grad(lambda p: jnp.real(model.astrometric_shift(times, p)).sum())(params)
    assert isinstance(derivative, sl.Parameters)
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(derivative))
    other = params.replace(t0=-0.03)
    parameter_batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params, other)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(model.astrometric_shift, in_axes=(None, 0)))(times, parameter_batch),
        jnp.stack([model.astrometric_shift(times, params), model.astrometric_shift(times, other)]),
        atol=1e-14,
    )


def test_moment_kernels_are_lazy_and_cold_model_jit_does_not_leak():
    model = sl.Model(source="uniform", fft=sl.FFTConfig(n_fft=128))
    assert "_astrometric_solver" not in model.__dict__
    model.magnification_at(0.02, rho=0.01)
    assert "_astrometric_solver" not in model.__dict__
    with jax.checking_leaks():
        actual = jax.jit(model.astrometric_shift_at)(0.02, rho=0.01)
    solver = model._astrometric_solver
    np.testing.assert_allclose(model.astrometric_shift_at(0.02, rho=0.01), actual, atol=1e-14)
    assert model._astrometric_solver is solver


@pytest.mark.parametrize("finite", [False, True])
def test_annual_parallax_reuses_the_trajectory_frame(finite):
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0, use_hjd=False)
    times = jnp.array([[7950.0, 8000.0, 8050.0]])
    kwargs = {"rho": 0.01, "a1": 0.5, "fft": sl.FFTConfig(n_fft=128)} if finite else {}

    def evaluate(pi):
        return sl.astrometric_shift(
            times, 8000.0, -0.05, 30.0, parallax=geometry, pi_e_n=pi[0], pi_e_e=pi[1], **kwargs
        )

    np.testing.assert_allclose(
        jax.jit(evaluate)(jnp.zeros(2)),
        sl.astrometric_shift(times, 8000.0, -0.05, 30.0, **kwargs),
        atol=1e-14,
    )
    pi = jnp.array([0.2, -0.1])
    result = evaluate(pi)
    if not finite:
        dtau, dbeta = geometry.offsets(times, pi_e_n=pi[0], pi_e_e=pi[1])
        position = (times - 8000) / 30 + dtau + 1j * (-0.05 + dbeta)
        np.testing.assert_allclose(result, point_source_shift(position), atol=1e-14)
    derivative = jax.jacfwd(evaluate)(pi)
    for i in range(2):
        offset = jnp.zeros(2).at[i].set(1e-5)
        np.testing.assert_allclose(
            derivative[..., i],
            (evaluate(pi + offset) - evaluate(pi - offset)) / 2e-5,
            rtol=1e-5,
            atol=1e-8,
        )
    assert result.shape == times.shape


def test_astrometry_parameter_validation():
    with pytest.raises(ValueError, match="rho or a1"):
        sl.astrometric_shift(1.0, 0.0, 0.1, 20.0, a1=0.5)
    with pytest.raises(ValueError, match="Parallax amplitudes"):
        sl.astrometric_shift(1.0, 0.0, 0.1, 20.0, pi_e_n=0.2)
    with pytest.raises(TypeError, match="Parameters"):
        sl.Model().astrometric_shift(1.0, {})
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=128))
    with pytest.raises(ValueError, match="require rho"):
        model.astrometric_shift_at(0.01, a1=0.5)
    with pytest.raises(ValueError, match="requires a1"):
        model.astrometric_shift_at(0.01, rho=0.01)
    with pytest.raises(ValueError, match="scalar"):
        model.astrometric_shift_at(0.01, rho=jnp.array([0.01]), a1=0.5)
    compiled = jax.jit(model.astrometric_shift_at)
    for a1 in (-0.1, 1.1, np.nan, np.inf):
        assert np.isnan(compiled(0.01 + 0.02j, rho=0.01, a1=a1))
    for rho in (-0.01, np.nan, np.inf):
        assert np.isnan(compiled(0.01 + 0.02j, rho=rho, a1=0.5))
    params = sl.Parameters(t0=0.0, u0=jnp.array([0.1, 0.2]), t_e=20.0, rho=0.01, a1=0.5)
    with pytest.raises(ValueError, match="vmap"):
        model.astrometric_shift(0.0, params)
