"""Interpolation/serialization and independent held-out magnification checks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens.lensing import BasisTable, pspl
from singlens.lensing.concentric import ConcentricDisk


class PolynomialLogBases:
    def basis(self, u, rho):
        x, y = jnp.log(rho), u / rho
        return jnp.atleast_1d(jnp.exp(0.3 * x + 0.01 * y**2)), jnp.atleast_1d(
            jnp.exp(-0.2 * x + 0.03 * y)
        )


@pytest.fixture(scope="module")
def polynomial_table():
    return BasisTable.build(PolynomialLogBases(), np.geomspace(0.01, 1, 8), np.linspace(0, 3, 11))


def test_table_values_and_gradients(polynomial_table):
    reference = PolynomialLogBases()
    u = jnp.array([[0.01, 0.04], [0.1, 0.2]])
    rho = 0.1
    for got, expected in zip(polynomial_table.basis(u, rho), reference.basis(u, rho)):
        np.testing.assert_allclose(got, expected, rtol=3e-14)
    for argnum in (0, 1):
        got = jax.jit(jax.jacfwd(polynomial_table.basis, argnums=argnum))(u, rho)
        expected = jax.jacfwd(reference.basis, argnums=argnum)(u, rho)
        for actual, ref in zip(got, expected):
            np.testing.assert_allclose(actual, ref, rtol=3e-12, atol=1e-13)
    # Closed spline endpoints must not silently extrapolate or select an empty span.
    for radius in (0.01, 1.0):
        got = polynomial_table.basis(jnp.array([0, 3 * radius]), radius)
        for actual, ref in zip(got, reference.basis(jnp.array([0, 3 * radius]), radius)):
            np.testing.assert_allclose(actual, ref, rtol=3e-14)
        for actual, ref in zip(
            jax.jacfwd(polynomial_table.basis, argnums=1)(0.0, radius),
            jax.jacfwd(reference.basis, argnums=1)(0.0, radius),
        ):
            np.testing.assert_allclose(actual, ref, rtol=3e-12)
        for actual, ref in zip(
            jax.jacfwd(polynomial_table.basis, argnums=0)(3 * radius, radius),
            jax.jacfwd(reference.basis, argnums=0)(3 * radius, radius),
        ):
            np.testing.assert_allclose(actual, ref, rtol=3e-12)


def test_table_domain_and_point_limit(polynomial_table):
    evaluate = jax.jit(polynomial_table.A)
    assert np.isnan(evaluate(0.1, 2.0)).all()
    assert np.isnan(evaluate(1.0, 0.1)).all()
    for radius in (-1, np.nan, np.inf):
        assert np.isnan(evaluate(0.1, radius)).all()
    for a1 in (-0.1, 1.1, 3, np.nan):
        assert np.isnan(evaluate(0.1, 0.1, a1=a1)).all()
    for a1 in (0, 0.5, 1):
        np.testing.assert_allclose(
            evaluate(jnp.array([0, 0.1]), 0.0, a1=a1), pspl(jnp.array([0, 0.1]))
        )
    with pytest.raises(ValueError, match="scalar"):
        polynomial_table.A(0.1, jnp.ones(2))


def test_table_round_trip(polynomial_table, tmp_path):
    path = tmp_path / "basis.npz"
    polynomial_table.save(path)
    restored = BasisTable.load(path)
    np.testing.assert_array_equal(
        restored.A(jnp.array([0, 0.01, 0.1]), 0.1),
        polynomial_table.A(jnp.array([0, 0.01, 0.1]), 0.1),
    )


def test_float32_round_trip_keeps_validated_host_knots(polynomial_table, tmp_path):
    source = tmp_path / "source.npz"
    saved = tmp_path / "float32.npz"
    polynomial_table.save(source)
    with jax.enable_x64(False):
        low_precision = BasisTable.load(source)
        assert low_precision._knots_rho.dtype == jnp.float32
        low_precision.save(saved)
        restored = BasisTable.load(saved)
        positions = jnp.array([0.0, 0.01, 0.1], dtype=jnp.float32)
        np.testing.assert_array_equal(restored.A(positions, 0.1), low_precision.A(positions, 0.1))
    # A float32 save must remain loadable after enabling x64, and preserve
    # the original domain/knots rather than changing the domain to rounded JAX
    # endpoints. The serialized coefficients retain the chosen precision.
    with np.load(source) as original, np.load(saved) as archive:
        for name in ("rho_grid", "z_grid", "knots_rho", "knots_z"):
            np.testing.assert_array_equal(archive[name], original[name])
        assert archive["coefficients"].dtype == np.float32
    with jax.enable_x64(True):
        restored = BasisTable.load(saved)
        assert restored._knots_rho.dtype == jnp.float64
        assert np.isfinite(restored.A(jnp.array([0.0, 0.01, 0.1]), 0.1)).all()


def test_physical_table_held_out_values():
    # Held-out values probe both grid axes rather than reproducing stored nodes.
    solver = ConcentricDisk(n_disk=32, n_limb=16)
    grid = np.unique(np.r_[np.linspace(0, 0.9, 70), 1.0, np.linspace(1.1, 3.0, 90)])
    table = BasisTable.build(solver, np.geomspace(0.01, 0.3, 20), grid)
    rho = 0.073
    u = rho * jnp.array([0.17, 0.53, 0.77, 1.53, 2.12])
    reference = ConcentricDisk(n_disk=96, n_limb=64)
    np.testing.assert_allclose(table.A(u, rho), reference.A(u, rho), rtol=2e-6)


@pytest.mark.parametrize("grid", [[0, 1, 2], [0, 1, 1, 2], [0, 1, 2, np.nan]])
def test_bad_grid(grid):
    with pytest.raises(ValueError):
        BasisTable.build(PolynomialLogBases(), np.geomspace(0.01, 1, 8), grid)
