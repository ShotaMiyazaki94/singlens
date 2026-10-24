"""Explicit event models under JIT, differentiation, and event batching."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import singlens as sl
from singlens.lensing import FFTConfig, pspl


@pytest.mark.parametrize("source", ["point", "uniform", "linear", "mu2"])
def test_model_first_jit_derivatives_and_event_batch(source):
    config = None if source == "point" else FFTConfig(n_fft=128)
    model = sl.Model(source=source, solver=config)
    times = jnp.array([-0.41, 0.07, 0.33])
    theta = jnp.array([0.01, 0.006, 20.0, 0.012, 0.4])

    def evaluate(p):
        params = sl.Parameters(
            t0=p[0],
            u0=p[1],
            t_e=p[2],
            rho=None if source == "point" else p[3],
            a1=p[4] if source == "linear" else None,
        )
        return model.magnification(times, params)

    # The first evaluation is compiled, with the model prepared outside JIT.
    with jax.checking_leaks():
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


@pytest.mark.parametrize("finite", [False, True])
def test_model_parallax_zero_limit_and_amplitude_derivatives(finite):
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0, use_hjd=False)
    times = jnp.array([[7950.0, 8000.0, 8050.0]])
    source = "linear" if finite else "point"
    config = FFTConfig(n_fft=128) if finite else None
    model = sl.Model(source=source, parallax=geometry, solver=config)
    static = sl.Model(source=source, solver=config)
    params = sl.Parameters(
        t0=8000.0,
        u0=0.05,
        t_e=30.0,
        rho=0.01 if finite else None,
        a1=0.5 if finite else None,
    )

    def evaluate(pi):
        return model.magnification(times, params.replace(pi_e_n=pi[0], pi_e_e=pi[1]))

    zero = jax.jit(evaluate)(jnp.zeros(2))
    np.testing.assert_allclose(zero, static.magnification(times, params))
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


def test_event_parameter_errors_and_scalar():
    model = sl.Model(source="point")
    params = sl.Parameters(t0=0.0, u0=0.1, t_e=20.0)
    assert jax.jit(model.magnification)(1.0, params).shape == ()
    with pytest.raises(ValueError, match="rho or a1"):
        model.magnification(1.0, params.replace(a1=0.5))
    with pytest.raises(ValueError, match="Parallax amplitudes"):
        model.magnification(1.0, params.replace(pi_e_n=0.2))


def test_static_point_source_against_analytic_curve():
    times = jnp.linspace(-10, 10, 21)
    expected = pspl(jnp.hypot(times / 20, 0.1))
    model = sl.Model(source="point")
    params = sl.Parameters(t0=0.0, u0=-0.1, t_e=20.0)
    np.testing.assert_allclose(model.magnification(times, params), expected)
    np.testing.assert_allclose(model.flux(times, params, f_s=2.0, f_b=0.3), 2 * expected + 0.3)
    np.testing.assert_allclose(model.trajectory(times, params), times / 20 - 0.1j)
    assert np.isnan(model.magnification(times, params.replace(t_e=-20.0))).all()
