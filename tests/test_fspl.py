from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from singlens.lensing import FFTConfig, LinearLimbDarkened, QuadraticProfile, UniformDisk, pspl


@lru_cache(None)
def reference(u, rho, profile):
    """Independent integration in lens-centered polar coordinates.

    The area Jacobian cancels the point-lens singularity. No FFT or singlens
    magnification is used in this reference.
    """

    def angle(theta):
        delta = np.sqrt(max(0.0, rho * rho - u * u * np.sin(theta) ** 2))
        lo, hi = max(0.0, u * np.cos(theta) - delta), u * np.cos(theta) + delta

        def radial(p):
            mu2 = max(0.0, 1 - (p * p + u * u - 2 * p * u * np.cos(theta)) / rho**2)
            intensity = {"disk": 1.0, "linear": 0.5 + 0.5 * np.sqrt(mu2), "quadratic": mu2}[profile]
            return intensity * (p * p + 2) / np.hypot(p, 2)

        # Integrate in units of rho so absolute tolerance stays meaningful
        # even for rho=1e-8. The missing radial rho is restored below.
        return quad(lambda x: radial(x * rho), lo / rho, hi / rho, epsabs=1e-12, epsrel=1e-9)[0]

    upper = np.pi if u <= rho else np.arcsin(rho / u)
    avg = {"disk": 1.0, "linear": 5 / 6, "quadratic": 1 / 2}[profile]
    integral = quad(angle, 0, upper, epsabs=1e-12, epsrel=1e-8)[0]
    return 2 * integral / (np.pi * rho * avg)


@pytest.fixture(
    scope="module",
    params=[
        (UniformDisk, "disk"),
        (LinearLimbDarkened, "linear"),
        (QuadraticProfile, "quadratic"),
    ],
)
def model(request):
    cls, name = request.param
    return cls(), name


@pytest.mark.parametrize("rho", [1e-6, 1e-5, 9.99e-5, 1e-4, 0.001, 0.01, 0.1, 1.0])
def test_independent_quadrature(model, rho):
    source, name = model
    ratios = np.array(
        [0.0, 0.00999, 0.01001, 0.1, 0.5, 0.9, 0.998, 1.0, 1.002, 1.1, 2.0, 10.0, 19.999, 20.001]
    )
    expected = [reference(float(r * rho), rho, name) for r in ratios]
    np.testing.assert_allclose(
        source.A(ratios * rho, rho), expected, rtol=1.1e-3 if name == "disk" else 1e-3
    )


@pytest.mark.parametrize("rho", [1e-6, 0.01, 1.0, 10.0])
def test_near_alignment_value_and_slope(model, rho):
    source, name = model
    ratios = np.array([1e-8, 1e-4, 0.005, 0.009])
    expected = [reference(float(r * rho), rho, name) for r in ratios]
    np.testing.assert_allclose(source.A(ratios * rho, rho), expected, rtol=1e-7)
    u, h = 0.005 * rho, 0.001 * rho
    slope = (reference(u + h, rho, name) - reference(u - h, rho, name)) / (2 * h)
    np.testing.assert_allclose(jax.grad(lambda x: source.A(x, rho).sum())(u), slope, rtol=2e-4)


@pytest.mark.parametrize("n_fft", [16384, 32768])
def test_refined_grid_resolves_central_transition(model, n_fft):
    original, name = model
    source = type(original)(fft=FFTConfig(n_fft=n_fft))
    rho = 1e-4
    ratios = np.array([0.00999, 0.01, 0.01001, 0.02, 0.1, 0.998, 1.002])
    expected = [reference(float(z * rho), rho, name) for z in ratios]
    np.testing.assert_allclose(source.A(ratios * rho, rho), expected, rtol=1e-3)


@pytest.mark.parametrize("rho", [1e-8, 1e-4, 0.01, 1.0, 10.0])
def test_center_and_fourier_normalization(model, rho):
    source, name = model
    np.testing.assert_allclose(source.A(0.0, rho)[0], reference(0.0, rho, name), rtol=1e-9)
    np.testing.assert_allclose(source.sk(jnp.array([0.0, 1e-9]), rho), 1.0, atol=1e-14)
    assert np.isfinite(jax.grad(lambda r: source.A(0.0, r).sum())(rho))
    assert jax.grad(lambda u: source.A(u, rho).sum())(0.0) == 0.0


def test_jit_vmap_and_gradients(model):
    source, _ = model
    u = jnp.array([0.0031, 0.0064, 0.0162, 0.039])
    rho = 0.0123
    compiled = jax.jit(source.A)
    np.testing.assert_allclose(compiled(u, rho), source.A(u, rho), rtol=1e-12)
    np.testing.assert_allclose(jax.vmap(lambda x: source.A(x, rho)[0])(u), compiled(u, rho))

    def f(r):
        return jnp.sum(source.A(u, r))

    step = 1e-7
    fd = (f(rho + step) - f(rho - step)) / (2 * step)
    np.testing.assert_allclose(jax.grad(f)(rho), fd, rtol=2e-5)
    np.testing.assert_allclose(jax.jacfwd(f)(rho), jax.grad(f)(rho), rtol=1e-11)
    assert np.isfinite(jax.vmap(lambda r: source.A(0.01, r)[0])(jnp.array([0.0, 1e-6, 0.01]))).all()


def test_gradients_against_independent_integral(model):
    """Finite differences of an independent integral, away from limb contact."""
    source, profile = model
    rho, step = 0.01, 1e-6
    derivative = jax.jit(jax.grad(lambda u, r: source.A(u, r)[0], argnums=(0, 1)))
    for ratio in (0.5, 1.5):
        u = ratio * rho
        du = (reference(u + step, rho, profile) - reference(u - step, rho, profile)) / (2 * step)
        dr = (reference(u, rho + step, profile) - reference(u, rho - step, profile)) / (2 * step)
        np.testing.assert_allclose(derivative(u, rho), [du, dr], rtol=1e-2)


def test_limits_shapes_and_invalid_inputs(model):
    source, _ = model
    u = np.array([[0.01, 0.02], [0.1, 0.2]])
    np.testing.assert_allclose(source.A(u, 0.0), pspl(u))
    assert source.A(0.1, 0.01).shape == (1,)
    assert source.A(u, 0.01).shape == (2, 2)
    assert np.isnan(jax.jit(source.A)(u, -0.1)).all()
    assert np.isnan(source.A(u, np.nan)).all()
    with pytest.raises(ValueError, match="scalar"):
        source.A(u, jnp.array([0.1, 0.2]))


@pytest.mark.parametrize("rho", [1e-6, 0.01, 0.1, 1.0, 10.0])
def test_far_field_against_independent_integral(model, rho):
    source, profile = model
    u = rho * np.array([20.0, 30.0, 50.0])
    expected = [reference(float(x), rho, profile) for x in u]
    np.testing.assert_allclose(jax.jit(source.A)(u, rho), expected, rtol=1e-6)


def test_entirely_far_batch_skips_fft(monkeypatch):
    source = UniformDisk()
    original = source._large_rho_grid
    calls = []

    def grid(rho, **profile):
        jax.debug.callback(lambda: calls.append("fft"))
        return original(rho, **profile)

    monkeypatch.setattr(source, "_large_rho_grid", grid)
    evaluate = jax.jit(source.A)
    evaluate(jnp.array([0.2, 0.3, 0.5]), 0.01).block_until_ready()
    jax.effects_barrier()
    assert calls == []
    derivative = jax.jit(jax.grad(lambda rho: source.A(jnp.array([0.2, 0.3, 0.5]), rho).sum()))
    derivative(0.01).block_until_ready()
    jax.effects_barrier()
    assert calls == []
    evaluate(jnp.array([0.005, 0.3, 0.5]), 0.01).block_until_ready()
    jax.effects_barrier()
    assert calls == ["fft"]


def test_far_values_and_gradients_do_not_depend_on_batch():
    source = LinearLimbDarkened()
    far = jnp.array([0.21, 0.4, 1.0])
    mixed = jnp.r_[0.005, far]

    def curve(theta, positions):
        return source.A(positions, theta[0], a1=theta[1])

    theta = jnp.array([0.01, 0.4])
    alone, together = curve(theta, far), curve(theta, mixed)[1:]
    np.testing.assert_allclose(alone, together, rtol=1e-13)
    derivative = jax.jit(jax.jacfwd(curve))
    alone, together = derivative(theta, far), derivative(theta, mixed)[1:]
    np.testing.assert_allclose(alone, together, rtol=1e-11, atol=1e-13)
    assert np.all(alone[:, 0] > 0)  # The radius correction is retained.
    assert np.all(alone[:, 1] < 0)  # Limb darkening changes the radial moment.


@pytest.mark.parametrize("ratio", [10.0, 20.0])
def test_far_transition_value_slope_and_curvature(model, ratio):
    source, _ = model
    rho = 0.01

    def curve(u):
        return source.A(u, rho)[0]

    values = jax.jit(jax.vmap(curve))
    slopes = jax.jit(jax.vmap(jax.grad(curve)))
    curvatures = jax.jit(jax.vmap(jax.grad(jax.grad(curve))))
    u = rho * ratio * jnp.array([1 - 1e-8, 1 + 1e-8])
    for evaluate in (values, slopes, curvatures):
        left, right = evaluate(u)
        assert np.isfinite([left, right]).all()
        np.testing.assert_allclose(left, right, rtol=1e-5, atol=1e-12)


def test_linear_coefficient_can_be_fitted():
    def f(a):
        return LinearLimbDarkened(a).A(jnp.array([0.0, 0.006]), 0.01).sum()

    a, h = 0.4, 1e-5
    np.testing.assert_allclose(jax.jit(jax.grad(f))(a), (f(a + h) - f(a - h)) / (2 * h), rtol=1e-7)
    np.testing.assert_allclose(LinearLimbDarkened(0).A(0.01, 0.02), UniformDisk().A(0.01, 0.02))


@pytest.mark.parametrize("rho", [1e-6, 0.01])
def test_uniform_derivatives_against_elliptic_limit(rho):
    """Check uniform derivatives in an explicit high-resolution calculation."""
    from scipy.special import ellipe, ellipk

    z = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 1.5, 2.0])
    inside = z < 1
    m = np.where(inside, z**2, 1 / z**2)
    e, k = ellipe(m), ellipk(m)
    first = 4 / np.pi * np.where(inside, (e - k) / z, e - k) / rho**2
    second = (
        4
        / np.pi
        * np.where(inside, -e / (1 - z**2) - (e - k) / z**2, e / (z * (z**2 - 1)))
        / rho**3
    )
    source = UniformDisk(fft=FFTConfig(n_fft=8192))

    def value(u):
        return source.A(u, rho)[0]

    np.testing.assert_allclose(jax.jit(jax.vmap(jax.grad(value)))(z * rho), first, rtol=1e-3)
    np.testing.assert_allclose(
        jax.jit(jax.vmap(jax.grad(jax.grad(value))))(z * rho), second, rtol=5e-3
    )


@pytest.mark.parametrize(
    "kwargs", [{"n_fft": 63}, {"n_fft": 65}, {"log_u_min": 4}, {"rho_switch": 0}]
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FFTConfig(**kwargs)
