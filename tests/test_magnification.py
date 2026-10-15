"""The function-first interface, including a cold first call under transformations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import singlens as sl
from singlens.lightcurve import _prepared_model


@pytest.mark.parametrize("source", ["point", "uniform", "linear"])
def test_function_cold_jit_derivatives_and_batch(source):
    _prepared_model.cache_clear()
    config = None if source == "point" else sl.FFTConfig(n_fft=128)
    t = jnp.array([-0.41, 0.07, 0.33])
    theta = jnp.array([0.01, 0.006, 20.0, 0.012, 0.4])

    def evaluate(p):
        return sl.magnification(
            t,
            *p[:3],
            rho=None if source == "point" else p[3],
            a1=p[4] if source == "linear" else None,
            fft=config,
        )

    # No eager preparation: a trace must not leak into the cached model.
    actual = jax.jit(evaluate)(theta)
    np.testing.assert_allclose(evaluate(theta), actual, rtol=1e-12)
    derivative = jax.jit(jax.jacfwd(evaluate))(theta)
    np.testing.assert_allclose(derivative, jax.jacrev(evaluate)(theta), rtol=1e-10, atol=1e-8)
    for i in range(5):
        step = 1e-5 * max(abs(float(theta[i])), 0.001)
        offset = jnp.zeros(5).at[i].set(step)
        finite_difference = (evaluate(theta + offset) - evaluate(theta - offset)) / (2 * step)
        np.testing.assert_allclose(derivative[:, i], finite_difference, rtol=2e-4, atol=1e-6)
    batch = jnp.stack([theta, theta.at[0].set(-0.03)])
    np.testing.assert_allclose(
        jax.jit(jax.vmap(evaluate))(batch), jnp.stack([evaluate(p) for p in batch])
    )
    model = sl.Model(source=source, fft=config)
    expected = model.magnification(
        t,
        sl.Parameters(
            t0=theta[0],
            u0=theta[1],
            t_e=theta[2],
            rho=theta[3] if source != "point" else None,
            a1=theta[4] if source == "linear" else None,
        ),
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12)


@pytest.mark.parametrize("finite", [False, True])
def test_parallax_function(finite):
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0, use_hjd=False)
    times = jnp.array([[7950.0, 8000.0, 8050.0]])
    kwargs = dict(rho=0.01, a1=0.5, fft=sl.FFTConfig(n_fft=128)) if finite else {}

    def evaluate(pi):
        return sl.magnification(
            times, 8000.0, 0.05, 30.0, parallax=geometry, pi_e_n=pi[0], pi_e_e=pi[1], **kwargs
        )

    zero = jax.jit(evaluate)(jnp.zeros(2))
    np.testing.assert_allclose(zero, sl.magnification(times, 8000.0, 0.05, 30.0, **kwargs))
    theta = jnp.array([0.2, -0.1])
    derivative = jax.jacfwd(evaluate)(theta)
    for i in range(2):
        delta = jnp.zeros(2).at[i].set(1e-5)
        np.testing.assert_allclose(
            derivative[..., i],
            (evaluate(theta + delta) - evaluate(theta - delta)) / 2e-5,
            rtol=1e-5,
            atol=1e-7,
        )
    assert evaluate(theta).shape == times.shape


def test_function_parameter_errors_and_scalar():
    assert sl.magnification(1.0, 0.0, 0.1, 20.0).shape == ()
    with pytest.raises(ValueError, match="rho or a1"):
        sl.magnification(1.0, 0.0, 0.1, 20.0, a1=0.5)
    with pytest.raises(ValueError, match="Parallax amplitudes"):
        sl.magnification(1.0, 0.0, 0.1, 20.0, pi_e_n=0.2)


def test_static_point_source_against_analytic_curve():
    times = jnp.linspace(-10, 10, 21)
    expected = sl.pspl(jnp.hypot(times / 20, 0.1))
    np.testing.assert_allclose(sl.magnification(times, 0.0, -0.1, 20.0), expected)
    model = sl.Model()
    params = sl.Parameters(t0=0.0, u0=-0.1, t_e=20.0)
    np.testing.assert_allclose(model.flux(times, params, f_s=2.0, f_b=0.3), 2 * expected + 0.3)
    np.testing.assert_allclose(model.trajectory(times, params), times / 20 - 0.1j)
    assert np.isnan(sl.magnification(times, 0.0, 0.1, -20.0)).all()
