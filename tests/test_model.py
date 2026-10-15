from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import singlens as sl


@pytest.fixture(scope="module", params=["point", "uniform", "linear", "mu2"])
def model_and_params(request):
    source = request.param
    model = sl.Model(source=source, fft=None if source == "point" else sl.FFTConfig(n_fft=128))
    p = sl.Parameters(
        t0=0.03,
        u0=0.006,
        t_e=20,
        rho=None if source == "point" else 0.0123,
        a1=0.3 if source == "linear" else None,
    )
    return model, p


def test_model_matches_source_profiles(model_and_params):
    model, params = model_and_params
    times = jnp.array([-0.81, -0.21, 0.03, 0.61])
    u = jnp.hypot((times - params.t0) / params.t_e, params.u0)
    if model.source == "point":
        expected = (u**2 + 2) / (u * jnp.sqrt(u**2 + 4))
    else:
        cls = {
            "uniform": sl.UniformDisk,
            "linear": sl.LinearLimbDarkened,
            "mu2": sl.QuadraticProfile,
        }[model.source]
        source = cls(
            fft=sl.FFTConfig(n_fft=128), **({"a1": 0.3} if model.source == "linear" else {})
        )
        expected = source.A(u, params.rho)
    np.testing.assert_allclose(model.magnification(times, params), expected, rtol=1e-11)
    np.testing.assert_allclose(model.flux(times, params, f_s=2.0, f_b=0.4), 2 * expected + 0.4)
    assert model.magnification(0.1, params).shape == ()
    assert model.magnification(times.reshape(2, 2), params).shape == (2, 2)
    position = model.trajectory(times, params)
    np.testing.assert_allclose(
        model.magnification_at(position, rho=params.rho, a1=params.a1), expected, rtol=1e-11
    )


def test_parameters_are_differentiable_without_unpacking(model_and_params):
    model, params = model_and_params
    times = jnp.array([-0.91, 0.34, 0.76])
    compiled = jax.jit(model.magnification)
    np.testing.assert_allclose(compiled(times, params), model.magnification(times, params))
    forward = jax.jit(jax.jacfwd(model.magnification, argnums=1))(times, params)
    reverse = jax.jacrev(model.magnification, argnums=1)(times, params)
    assert isinstance(forward, sl.Parameters)
    assert forward.t_e.shape == times.shape
    for a, b in zip(jax.tree_util.tree_leaves(forward), jax.tree_util.tree_leaves(reverse)):
        assert np.isfinite(a).all()
        np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-9)
    if model.source == "linear":
        h = 1e-5
        fd = (
            compiled(times, params.replace(a1=params.a1 + h))
            - compiled(times, params.replace(a1=params.a1 - h))
        ) / (2 * h)
        np.testing.assert_allclose(forward.a1, fd, rtol=1e-8)


def test_vmap_and_immutable_parameters():
    model = sl.Model()
    params = sl.Parameters(t0=0, u0=0.1, t_e=20)
    assert params.t_e.dtype.kind == "f"
    with pytest.raises(FrozenInstanceError):
        params.t_e = 10
    updated = params.replace(u0=0.2)
    assert float(params.u0) == 0.1
    assert float(updated.u0) == 0.2
    batch = jax.tree_util.tree_map(lambda a, b: jnp.stack([a, b]), params, updated)
    times = jnp.array([-1.0, 0.0, 1.0])
    actual = jax.jit(jax.vmap(model.magnification, in_axes=(None, 0)))(times, batch)
    np.testing.assert_allclose(
        actual, jnp.stack([model.magnification(times, params), model.magnification(times, updated)])
    )
    with pytest.raises(ValueError, match="vmap"):
        model.magnification(times, batch)
    with pytest.raises(ValueError, match="real"):
        sl.Parameters(t0=1j, u0=0.1, t_e=20)
    with pytest.raises(ValueError, match="required"):
        sl.Parameters(t0=None, u0=0.1, t_e=20)


def test_named_flux_fit_and_jit():
    model = sl.Model()
    params = sl.Parameters(t0=0, u0=0.1, t_e=20)
    times = jnp.array([-20.0, -5.0, 0.0, 10.0])
    observed = model.flux(times, params, f_s=1.7, f_b=0.4) + jnp.array([0.01, -0.02, 0.03, -0.01])
    result = jax.jit(model.fit_flux)(times, params, observed, 0.1)
    mag = model.magnification(times, params)
    expected = sl.fit_flux(mag, observed, 0.1)
    for actual, reference in zip(result, expected):
        np.testing.assert_allclose(actual, reference)
    np.testing.assert_allclose(result.model_flux, result.f_s * mag + result.f_b)
    grad = jax.grad(lambda p: model.fit_flux(times, p, observed, 0.1).chi2)(params)
    assert np.isfinite(grad.t_e)
    with pytest.raises(ValueError, match="same shape"):
        model.fit_flux(times, params, observed[:-1], 0.1)
    with pytest.raises(ValueError, match="flux_err"):
        model.fit_flux(times, params, observed, jnp.ones((4, 1)))
    with pytest.raises(ValueError, match="scalar"):
        model.flux(times, params, f_s=jnp.ones(4))
    with pytest.raises(ValueError, match="real"):
        sl.fit_flux(mag, observed + 1j, 0.1)


@pytest.mark.parametrize("error", [0.0, -0.1, np.nan, np.inf])
def test_new_flux_fit_rejects_invalid_numeric_errors(error):
    result = jax.jit(sl.fit_flux)(jnp.array([1.0, 2.0, 3.0]), jnp.array([2.0, 3.0, 4.0]), error)
    assert all(np.isnan(x).all() for x in result)


def test_degenerate_flux_fit():
    assert all(np.isnan(x).all() for x in sl.fit_flux(jnp.ones(3), jnp.ones(3), 0.1))
    with pytest.raises(ValueError, match="at least two"):
        sl.fit_flux([1.0], [1.0], 0.1)


def test_model_options_and_missing_parameters():
    with pytest.raises(ValueError, match="source must"):
        sl.Model(source="binary")
    with pytest.raises(ValueError, match="finite-source"):
        sl.Model(fft=sl.FFTConfig())
    with pytest.raises(TypeError, match="FFTConfig"):
        sl.Model(source="uniform", fft={})
    with pytest.raises(TypeError, match="AnnualParallax"):
        sl.Model(parallax=())
    with pytest.raises(ValueError, match="n_fft"):
        sl.FFTConfig(n_fft=127)
    p = sl.Parameters(t0=0, u0=0.1, t_e=20)
    with pytest.raises(TypeError, match="Parameters"):
        sl.Model().magnification(0.0, {})
    with pytest.raises(ValueError, match="finite source"):
        sl.Model().magnification(0.0, p.replace(rho=0.01))
    finite = sl.Model(source="uniform", fft=sl.FFTConfig(n_fft=128))
    with pytest.raises(ValueError, match="require rho"):
        finite.magnification(0.0, p)
    with pytest.raises(ValueError, match="only to source"):
        finite.magnification(0.0, p.replace(rho=0.01, a1=0.5))
    with pytest.raises(ValueError, match="Parallax amplitudes"):
        sl.Model().magnification(0.0, p.replace(pi_e_n=0.2))
    assert np.isnan(sl.Model().magnification(0.0, p.replace(t_e=-20)))


def test_linear_boundaries_and_small_sources():
    model = sl.Model(source="linear", fft=sl.FFTConfig(n_fft=128))
    with pytest.raises(ValueError, match="requires a1"):
        model.magnification_at(0.01, rho=0.01)
    compiled = jax.jit(model.magnification_at)
    for a1 in [0.0, 1.0]:
        assert np.isinf(compiled(0.0, rho=0.0, a1=a1))
    for a1 in [-0.1, 1.1, np.nan]:
        assert np.isnan(compiled(0.01, rho=0.01, a1=a1))
    reference = sl.LinearLimbDarkened(0.4, fft=sl.FFTConfig(n_fft=128))
    u = jnp.array([0.0, 0.5e-6, 1e-6, 20e-6])
    np.testing.assert_allclose(compiled(u, rho=1e-6, a1=0.4), reference.A(u, 1e-6), rtol=1e-11)


def test_annual_parallax_explicit_time_origin():
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0, use_hjd=False)
    model = sl.Model(parallax=geometry)
    params = sl.Parameters(t0=8000.0, u0=0.1, t_e=30.0, pi_e_n=0.2, pi_e_e=-0.1)
    times = jnp.array([7950.0, 8000.0, 8050.0])
    dtau, dbeta = geometry.offsets(times, pi_e_n=0.2, pi_e_e=-0.1)
    expected = (times - params.t0) / params.t_e + dtau + 1j * (params.u0 + dbeta)
    np.testing.assert_allclose(model.trajectory(times, params), expected, atol=1e-14)
    assert geometry.offsets(8000.0, pi_e_n=0.2, pi_e_e=-0.1)[0].shape == ()
    absolute = sl.Model(
        parallax=sl.AnnualParallax(
            ra_deg=266.0, dec_deg=-29.0, t_ref=2458000.0, time_origin=0.0, use_hjd=False
        )
    )
    np.testing.assert_allclose(
        model.magnification(times, params),
        absolute.magnification(times + 2450000.0, params.replace(t0=2458000.0)),
    )
    derivative = jax.jit(jax.jacfwd(model.magnification, argnums=1))(times, params)
    assert np.isfinite(derivative.pi_e_n).all()
    with pytest.raises(ValueError, match="both pi_e_n"):
        model.magnification(times, params.replace(pi_e_e=None))
    with pytest.raises(FrozenInstanceError):
        model.source = "linear"
