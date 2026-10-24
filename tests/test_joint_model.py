"""Prepared-model integration of static photometric/astrometric backends."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens import Model, Parameters
from singlens.lensing import BasisTable, FFTConfig
from singlens.lensing.concentric import ConcentricDisk
from singlens.lensing.gould import GouldYoo


@pytest.fixture(scope="module")
def solvers():
    direct = ConcentricDisk(n_disk=32, n_limb=16)
    table = BasisTable.build(
        direct, np.geomspace(0.01, 1, 12), np.linspace(0, 3, 70), a1=0.9, astrometry=True
    )
    return FFTConfig(), table


@pytest.mark.parametrize("source", ["uniform", "linear"])
def test_static_backends_share_joint_and_separate_results(solvers, source):
    times = jnp.linspace(-0.12, 0.12, 9)
    p = Parameters(t0=0.0, u0=0.03, t_e=2.0, rho=0.1, a1=0.6 if source == "linear" else None)
    for solver in solvers:
        model = Model(source=source, solver=solver)
        joint = jax.jit(model.joint)
        flux, shift = joint(times, p)
        np.testing.assert_allclose(flux, model.magnification(times, p), rtol=2e-13)
        np.testing.assert_allclose(shift, model.astrometric_shift(times, p), rtol=2e-13)
        assert flux.shape == shift.shape == times.shape
        grad = jax.grad(
            lambda q: jnp.sum(model.joint(times, q)[0] + jnp.real(model.joint(times, q)[1]))
        )(p)
        for leaf in jax.tree.leaves(grad):
            assert np.all(np.isfinite(leaf))
        if isinstance(solver, BasisTable):
            center = jax.jit(model.joint_at)(0j, rho=0.1, a1=p.a1)
            assert center[0].shape == center[1].shape == ()
            assert np.isfinite(center[0]) and center[1] == 0


def test_uniform_profile_overrides_table_coefficient(solvers):
    table = solvers[1]
    model = Model(source="uniform", solver=table)
    position = jnp.array([0.01, 0.04, 0.101, 0.2])
    expected = table.joint(position, 0.1, a1=0.0)
    for actual, ref in zip(model.joint_at(position, rho=0.1), expected):
        np.testing.assert_allclose(actual, ref, rtol=3e-13, atol=1e-14)
    assert not np.allclose(expected[0], table.joint(position, 0.1)[0])


def test_point_model_joint_is_analytic():
    model = Model(source="point")
    position = jnp.array([0.01 + 0.2j, 3 + 4j])
    flux, shift = jax.jit(model.joint_at)(position)
    np.testing.assert_array_equal(flux, model.magnification_at(position))
    np.testing.assert_array_equal(shift, model.astrometric_shift_at(position))


def test_backend_validation_and_gould_photometry_only(solvers):
    table = solvers[1]
    for solver in solvers:
        with pytest.raises(ValueError, match="finite-source"):
            Model(source="point", solver=solver)
    with pytest.raises(ValueError, match="uniform.*linear"):
        Model(source="mu2", solver=table)
    with pytest.raises(TypeError, match="solver"):
        Model(source="linear", solver=ConcentricDisk())
    with pytest.raises(ValueError, match="solver"):
        Model(source="linear", solver="auto")
    with pytest.raises(TypeError, match="solver"):
        Model(source="linear", solver=GouldYoo())
    for method in ("joint", "shift", "joint_basis"):
        with pytest.raises(NotImplementedError, match="photometry only"):
            getattr(GouldYoo(), method)(0.1, 0.1)
    model = Model(source="linear", solver=table)
    for coefficient in (-1.0, 3.0, np.nan):
        result = model.joint_at(0.1, rho=0.1, a1=coefficient)
        assert all(np.isnan(value) for value in result)
