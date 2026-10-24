"""Packaged default tables: loading, numerical behavior, and JAX reuse."""

from importlib.resources import files

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import singlens as sl
from singlens.lensing import BasisTable, FFTConfig
from singlens.lensing.basis_table import _bundled_basis_table
from singlens.lensing.concentric import ConcentricDisk
from singlens.lensing.point import point_source_shift, pspl
from tools.validation.astrometry_accuracy import reference_shift


def test_packaged_resource_is_loaded_without_runtime_generation(monkeypatch):
    resource = files("singlens.data").joinpath("finite_source_table.npz")
    assert resource.is_file()
    with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as saved:
        coefficients = saved["coefficients"].copy()
        assert coefficients.shape == (192, 947, 4)
        assert coefficients.dtype == np.float64
        assert str(saved["representation"]) == "point_residual"

    def forbidden_generation(*args, **kwargs):
        raise AssertionError("Constructing a Model must not generate the bundled table")

    monkeypatch.setattr(BasisTable, "build", forbidden_generation)
    _bundled_basis_table.cache_clear()
    uniform = sl.Model(source="uniform")
    linear = sl.Model(source="linear")
    assert isinstance(uniform.solver, BasisTable)
    assert uniform.solver is linear.solver
    assert uniform.solver.has_astrometry
    assert uniform.solver.coordinate == "log1p_ratio2"
    assert uniform.solver.representation == "point_residual"
    np.testing.assert_array_equal(uniform.solver._coefficients, coefficients)
    np.testing.assert_array_equal(uniform.solver.rho_grid[[0, -1]], [1e-6, 10.0])
    np.testing.assert_array_equal(uniform.solver.z_grid[[0, -1]], [0.0, 1e8])


def test_point_and_mu2_defaults_and_explicit_fft():
    point = sl.Model(source="point")
    assert point.solver is None
    position = jnp.array([0.1 + 0.2j, 2 - 3j])
    flux, shift = point.joint_at(position)
    np.testing.assert_allclose(flux, pspl(position), rtol=1e-14)
    np.testing.assert_allclose(shift, point_source_shift(position), rtol=1e-14)

    mu2 = sl.Model(source="mu2")
    assert isinstance(mu2.solver, FFTConfig)
    assert mu2.solver.n_fft == 4096
    explicit_mu2 = sl.Model(source="mu2", solver="fft")
    for actual, expected in zip(
        mu2.joint_at(position, rho=0.1), explicit_mu2.joint_at(position, rho=0.1)
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-14)

    config = FFTConfig(n_fft=128)
    for source in ("uniform", "linear"):
        model = sl.Model(source=source, solver=config)
        assert model.solver is config
        coefficient = 0.6 if source == "linear" else None
        # This radius is deliberately below the packaged table's domain.
        assert np.isfinite(model.magnification_at(0.01, rho=1e-8, a1=coefficient))


@pytest.mark.parametrize("source", ["uniform", "linear"])
def test_explicit_table_selection_reuses_the_default(source):
    implicit = sl.Model(source=source)
    explicit = sl.Model(source=source, solver="table")
    assert isinstance(explicit.solver, BasisTable)
    assert explicit.solver is implicit.solver
    coefficient = 0.6 if source == "linear" else None
    position = jnp.array([0.03, 0.2])
    for actual, expected in zip(
        jax.jit(explicit.joint_at)(position, rho=0.1, a1=coefficient),
        implicit.joint_at(position, rho=0.1, a1=coefficient),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-13)


@pytest.mark.parametrize("source", ["uniform", "linear", "mu2"])
def test_fft_selection_uses_fixed_defaults_and_matches_expert_config(source):
    selected = sl.Model(source=source, solver="fft")
    config = FFTConfig()
    assert selected.solver == config
    assert selected.solver.n_fft == 4096
    expert = sl.Model(source=source, solver=config)
    assert expert.solver is config
    coefficient = 0.6 if source == "linear" else None
    position = jnp.array([0.03, 0.2])
    for actual, expected in zip(
        selected.joint_at(position, rho=0.1, a1=coefficient),
        expert.joint_at(position, rho=0.1, a1=coefficient),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-14)


@pytest.mark.parametrize("solver", ["table", "fft"])
def test_point_rejects_finite_source_solver_names(solver):
    with pytest.raises(ValueError, match="finite-source"):
        sl.Model(source="point", solver=solver)


def test_unsupported_table_profile_and_solver_values():
    with pytest.raises(ValueError, match="uniform.*linear"):
        sl.Model(source="mu2", solver="table")
    for solver in ("auto", "FFT", ""):
        with pytest.raises(ValueError, match="solver"):
            sl.Model(source="linear", solver=solver)
    for solver in ({}, [], 4096):
        with pytest.raises(TypeError, match="solver"):
            sl.Model(source="linear", solver=solver)


@pytest.mark.parametrize("source", ["uniform", "linear"])
def test_default_joint_separate_shapes_and_event_batch(source):
    model = sl.Model(source=source)
    times = jnp.array([[-0.13, 0.0, 0.11], [0.2, 0.8, 3.0]])
    params = sl.Parameters(
        t0=0.01, u0=-0.03, t_e=2.0, rho=0.1, a1=0.6 if source == "linear" else None
    )
    with jax.checking_leaks():
        flux, shift = jax.jit(model.joint)(times, params)
    assert flux.shape == shift.shape == times.shape
    np.testing.assert_allclose(flux, model.magnification(times, params), rtol=2e-13)
    np.testing.assert_allclose(shift, model.astrometric_shift(times, params), rtol=2e-13)
    scalar = model.joint(0.0, params)
    assert scalar[0].shape == scalar[1].shape == ()
    other = params.replace(t0=-0.03)
    batch = jax.tree.map(lambda a, b: jnp.stack([a, b]), params, other)
    actual = jax.jit(jax.vmap(model.joint, in_axes=(None, 0)))(times, batch)
    eager = [model.joint(times, p) for p in (params, other)]
    for component in (0, 1):
        np.testing.assert_allclose(actual[component], jnp.stack([v[component] for v in eager]))
    derivative = jax.grad(lambda p: jnp.sum(jnp.log(model.joint(times, p)[0])))(params)
    assert all(np.isfinite(v).all() for v in jax.tree.leaves(derivative))


def test_cold_cache_inside_jit_does_not_retain_tracers():
    _bundled_basis_table.cache_clear()

    def evaluate(position):
        # Stress a cold resource load while tracing, then reuse it eagerly.
        return sl.Model(source="linear").joint_at(position, rho=0.1, a1=0.6)

    with jax.checking_leaks():
        compiled = jax.jit(evaluate)(jnp.array([0.03, 0.13]))
    model = sl.Model(source="linear")
    assert not isinstance(model.solver._coefficients, jax.core.Tracer)
    for actual, expected in zip(compiled, model.joint_at(jnp.array([0.03, 0.13]), rho=0.1, a1=0.6)):
        np.testing.assert_allclose(actual, expected, rtol=2e-13)


def test_cache_keeps_float32_and_float64_coefficients_separate():
    _bundled_basis_table.cache_clear()
    with jax.enable_x64(False):
        low = sl.Model(source="uniform").solver
        assert low._coefficients.dtype == jnp.float32
    with jax.enable_x64(True):
        high = sl.Model(source="linear").solver
        assert high._coefficients.dtype == jnp.float64
        assert high is not low
        resource = files("singlens.data").joinpath("finite_source_table.npz")
        with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as saved:
            np.testing.assert_array_equal(high._coefficients, saved["coefficients"])
        # Enabling x64 must reload full precision, not upcast cached float32 data.
        assert np.any(np.asarray(high._coefficients) != np.asarray(low._coefficients, dtype=float))
    with jax.enable_x64(False):
        assert sl.Model(source="linear").solver is low
    with jax.enable_x64(True):
        assert sl.Model(source="uniform").solver is high
    assert _bundled_basis_table.cache_info().currsize == 2


@pytest.mark.parametrize("source", ["uniform", "linear"])
def test_default_domain_edges_point_limit_and_no_fallback(source):
    model = sl.Model(source=source)
    coefficient = 0.6 if source == "linear" else None
    evaluate = jax.jit(lambda u, r: model.joint_at(u, rho=r, a1=coefficient))
    for radius in (1e-6, 10.0):
        flux, shift = evaluate(radius * jnp.array([0.0, 1.0, 1e8]), radius)
        assert np.isfinite(flux).all() and np.isfinite(shift).all()
    for u, radius in ((0.01, 0.999e-6), (1.0, 10.001), (1.00001e8, 1.0)):
        flux, shift = evaluate(u, radius)
        assert np.isnan(flux) and np.isnan(shift)
    for radius in (-1.0, np.nan, np.inf):
        assert all(np.isnan(v) for v in evaluate(0.1, radius))
    positions = jnp.array([0.0, -0.1, 0.2 + 0.3j, 1e200])
    flux, shift = evaluate(positions, 0.0)
    np.testing.assert_allclose(flux, pspl(positions), rtol=1e-14)
    np.testing.assert_allclose(shift, point_source_shift(positions), rtol=1e-14, atol=0)
    np.testing.assert_allclose(jax.grad(lambda u: evaluate(u, 0.0)[1])(0.0), 0.5)


def test_closed_ratio_endpoint_survives_coordinate_roundoff():
    model = sl.Model(source="uniform")
    radii = jnp.asarray(np.r_[1.1377741332214902e-6, np.geomspace(1e-6, 10.0, 31)])
    directions = jnp.exp(1j * jnp.array([0.0, 0.37, 1.2]))

    def evaluate(radius, ratio):
        return model.joint_at(radius * ratio * directions, rho=radius)

    batched = jax.jit(jax.vmap(evaluate, in_axes=(0, None)))
    # Different radii/directions round the squared ratio to either side of
    # the same mathematical endpoint. All belong to the closed table domain.
    for value in batched(radii, 1e8):
        assert np.isfinite(value).all()
    for value in batched(radii, 1e8 * (1 + 1e-10)):
        assert np.isnan(value).all()


@pytest.mark.parametrize("rho", [0.073, 0.73])
def test_packaged_values_against_direct_and_independent_moments(rho):
    direct = ConcentricDisk(n_disk=96, n_limb=64)
    positions = rho * np.array([0.0, 0.53, 0.999, 1.0, 1.001, 2.13, 30.0])
    for source, coefficient in (("uniform", 0.0), ("linear", 0.6)):
        model = sl.Model(source=source)
        flux, shift = model.joint_at(
            positions, rho=rho, a1=coefficient if source == "linear" else None
        )
        expected_flux = direct.A(positions, rho, a1=coefficient)
        # This reference integrates image moments without a disk mixture/table.
        expected_shift = [reference_shift(u, rho, source, coefficient) for u in positions]
        np.testing.assert_allclose(flux, expected_flux, rtol=3e-7)
        np.testing.assert_allclose(shift, expected_shift, rtol=0, atol=2e-7)


def test_packaged_gradients_and_uniform_central_identity():
    model = sl.Model(source="linear")
    direct = ConcentricDisk(n_disk=96, n_limb=64)
    positions = jnp.array([0.53, 0.999, 1.001, 2.13]) * 0.73

    def vector(parameters, evaluate):
        flux, shift = evaluate(parameters[0] + 1j * parameters[1], parameters[2], parameters[3])
        return jnp.stack((flux, shift.real, shift.imag), axis=-1)

    def approximate(position, rho, a1):
        return model.joint_at(position, rho=rho, a1=a1)

    def reference(position, rho, a1):
        return direct.joint(position, rho, a1=a1)

    actual_grad = jax.jit(jax.jacfwd(lambda p: vector(p, approximate)))
    reference_grad = jax.jit(jax.jacfwd(lambda p: vector(p, reference)))
    for u in positions:
        theta = jnp.array([u, 0.0, 0.73, 0.6])
        actual, expected = actual_grad(theta), reference_grad(theta)
        scale = jnp.array([theta[2], theta[2], theta[2], 1.0])
        flux = approximate(u, theta[2], theta[3])[0]
        error = np.abs((actual - expected) * scale)
        assert np.max(error[0] / flux) < 2e-4
        assert np.max(error[1:]) < 2e-4

    uniform = sl.Model(source="uniform")
    rho = 0.73  # An interior held-out radius, not a generating knot.
    flux, shift = uniform.joint_at(0.0, rho=rho)
    np.testing.assert_allclose(flux, np.hypot(rho, 2) / rho, rtol=3e-7)
    assert shift == 0
    slope = jax.grad(lambda u: uniform.astrometric_shift_at(u, rho=rho))(0.0)
    np.testing.assert_allclose(slope, -1 / (rho**2 + 4), rtol=3e-7)
