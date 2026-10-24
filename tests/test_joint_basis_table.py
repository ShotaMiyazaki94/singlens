"""Signed astrometric spline channels, physical moments, and saved-table reuse."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens.lensing.basis_table import BasisTable
from singlens.lensing.concentric import ConcentricDisk
from singlens.lensing.point import point_source_shift, pspl


class PolynomialJointBases:
    """Exact cubic residuals in the documented even table coordinates."""

    def joint_basis(self, u, rho):
        x, y = jnp.log(rho), jnp.log1p((u / rho) ** 2)
        q = 1 + (u / rho) ** 2
        baseline_flux = pspl(jnp.hypot(u, rho))
        baseline_coefficient = 1 / (u**2 + 2)
        return tuple(
            jnp.atleast_1d(value)
            for value in (
                baseline_flux * jnp.exp((0.3 * x + 0.02 * y**2) / q),
                baseline_flux * jnp.exp((-0.2 * x + 0.03 * y) / q),
                baseline_coefficient + (-0.2 + 0.03 * x + 0.02 * y**2) / q,
                baseline_coefficient + (0.05 - 0.01 * x + 0.04 * y) / q,
            )
        )

    def basis(self, u, rho):
        return self.joint_basis(u, rho)[:2]


@pytest.fixture(scope="module")
def polynomial_joint_table():
    return BasisTable.build(
        PolynomialJointBases(),
        np.geomspace(0.01, 1, 8),
        np.linspace(0, 5, 16),
        astrometry=True,
    )


def expected_joint(position, rho, a1):
    source = PolynomialJointBases()
    f0, fm, c0, cm = source.joint_basis(jnp.abs(position), rho)
    gamma = 2 * a1 / (3 - a1)
    flux = (1 - gamma) * f0 + gamma * fm
    shift = position * ((1 - gamma) * f0 * c0 + gamma * fm * cm) / flux
    return flux.reshape(jnp.shape(position)), shift.reshape(jnp.shape(position))


def test_four_channels_reproduce_polynomials_and_signed_coefficients(polynomial_joint_table):
    assert polynomial_joint_table.has_astrometry
    u, rho = jnp.array([[0.0, 0.03], [0.11, 0.32]]), 0.1
    reference = PolynomialJointBases()
    for got, expected in zip(
        polynomial_joint_table.joint_basis(u, rho), reference.joint_basis(u, rho)
    ):
        np.testing.assert_allclose(got, expected, rtol=5e-13, atol=2e-14)
    for argnum in (0, 1):
        actual = jax.jit(jax.jacfwd(polynomial_joint_table.joint_basis, argnums=argnum))(u, rho)
        expected = jax.jacfwd(reference.joint_basis, argnums=argnum)(u, rho)
        for got, ref in zip(actual, expected):
            np.testing.assert_allclose(got, ref, rtol=3e-11, atol=3e-12)


def test_joint_moment_mixture_shapes_rotation_and_parameter_gradients(polynomial_joint_table):
    position = jnp.array([[0.0, -0.03], [0.1 + 0.07j, -0.2j]])
    theta = jnp.array([0.1, 0.43])

    def evaluate(p):
        flux, shift = polynomial_joint_table.joint(position, p[0], a1=p[1])
        return flux, shift.real, shift.imag

    def reference(p):
        flux, shift = expected_joint(position, p[0], p[1])
        return flux, shift.real, shift.imag

    for actual, expected in zip(jax.jit(evaluate)(theta), reference(theta)):
        np.testing.assert_allclose(actual, expected, rtol=5e-13, atol=2e-14)
        assert actual.shape == position.shape
    for actual, expected in zip(jax.jacfwd(evaluate)(theta), jax.jacfwd(reference)(theta)):
        np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=3e-12)
    rotation = np.exp(0.71j)
    np.testing.assert_allclose(
        polynomial_joint_table.shift(rotation * position, theta[0], a1=theta[1]),
        rotation * polynomial_joint_table.shift(position, theta[0], a1=theta[1]),
        rtol=3e-13,
        atol=2e-14,
    )
    assert polynomial_joint_table.shift(0.1, 0.1).shape == ()
    assert polynomial_joint_table.joint(0.1, 0.1)[0].shape == ()
    assert polynomial_joint_table.A(0.1, 0.1).shape == (1,)


def test_even_center_and_cartesian_jacobian(polynomial_joint_table):
    rho, coefficient = 0.1, 0.43

    def vector(xy):
        flux, shift = polynomial_joint_table.joint(xy[0] + 1j * xy[1], rho, a1=coefficient)
        return jnp.array([flux, shift.real, shift.imag])

    actual = jax.jit(jax.jacfwd(vector))(jnp.zeros(2))
    f0, fm, c0, cm = PolynomialJointBases().joint_basis(0.0, rho)
    gamma = 2 * coefficient / (3 - coefficient)
    slope = float(((1 - gamma) * f0 * c0 + gamma * fm * cm)[0] / ((1 - gamma) * f0 + gamma * fm)[0])
    expected = np.array([[0, 0], [slope, 0], [0, slope]])
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=1e-13)
    np.testing.assert_allclose(jax.jacrev(vector)(jnp.zeros(2)), actual, rtol=3e-12, atol=1e-13)
    # An even coordinate also keeps the limiting slope continuous near zero.
    for separation in (1e-8 * rho, 1e-5 * rho):
        np.testing.assert_allclose(
            polynomial_joint_table.shift(separation, rho, a1=coefficient) / separation,
            slope,
            rtol=2e-9,
        )


@pytest.fixture(scope="module")
def physical_joint_table():
    offsets = np.geomspace(1e-6, 0.2, 45)
    z_grid = np.unique(
        np.r_[np.linspace(0, 0.8, 65), 1 - offsets, 1, 1 + offsets, np.linspace(1.2, 4, 80)]
    )
    return BasisTable.build(
        ConcentricDisk(n_disk=64, n_limb=32),
        np.geomspace(0.02, 2, 32),
        z_grid,
        astrometry=True,
    )


@pytest.mark.parametrize("rho", [0.073, 0.73])
def test_physical_table_held_out_values_and_gradients(physical_joint_table, rho):
    reference = ConcentricDisk(n_disk=96, n_limb=64)
    ratios = jnp.array([0.0, 0.0017, 0.17, 0.53, 0.777, 0.999, 1.001, 1.531, 3.123])
    positions, coefficient = ratios * rho, 0.43
    flux, shift = physical_joint_table.joint(positions, rho, a1=coefficient)
    expected_flux, expected_shift = reference.joint(positions, rho, a1=coefficient)
    np.testing.assert_allclose(flux, expected_flux, rtol=3e-6)
    np.testing.assert_allclose(shift, expected_shift, rtol=0, atol=2e-6 * rho)
    # Both grid axes are held out. Derivatives at exact contact are singular.
    for ratio in (0.0, 0.17, 0.777, 1.531):
        theta = jnp.array([ratio * rho, rho, coefficient])

        def evaluate(p, solver):
            return solver.shift(p[0], p[1], a1=p[2])

        got = jax.grad(lambda p: evaluate(p, physical_joint_table))(theta)
        expected = jax.grad(lambda p: evaluate(p, reference))(theta)
        np.testing.assert_allclose(got, expected, rtol=2e-4, atol=2e-6)


def test_joint_table_roundtrip_and_domains(polynomial_joint_table, tmp_path):
    path = tmp_path / "joint-bases.npz"
    polynomial_joint_table.save(path)
    restored = BasisTable.load(path)
    assert restored.has_astrometry
    positions = jnp.array([[0.0, -0.01], [0.1 + 0.1j, -0.2j]])
    for expected, actual in zip(
        polynomial_joint_table.joint(positions, 0.1), restored.joint(positions, 0.1)
    ):
        np.testing.assert_array_equal(actual, expected)
    for radius in (-0.1, np.nan, np.inf, 0.001, 2.0):
        flux, shift = restored.joint(positions, radius)
        assert np.isnan(flux).all() and np.isnan(shift).all()
    for coefficient in (-0.1, 1.1, np.nan):
        flux, shift = restored.joint(positions, 0.1, a1=coefficient)
        assert np.isnan(flux).all() and np.isnan(shift).all()
    assert np.isnan(restored.shift(0.6, 0.1))
    with pytest.raises(ValueError, match="scalar"):
        restored.shift(0.1, jnp.array([0.1, 0.2]))
    with pytest.raises(ValueError, match="real"):
        restored.shift(0.1, 0.1 + 0.01j)
    for coefficient in (0.0, 0.5, 1.0):
        flux, shift = jax.jit(restored.joint)(positions, 0.0, a1=coefficient)
        np.testing.assert_allclose(flux, pspl(positions), rtol=1e-14)
        np.testing.assert_allclose(shift, point_source_shift(positions), rtol=1e-14)
    np.testing.assert_allclose(jax.grad(lambda x: restored.shift(x, 0.0))(0.0), 0.5, rtol=1e-14)


def test_joint_table_preserves_the_astrometric_wing():
    # The photometric excess is effectively zero here, but the image moment
    # still has a measurable 1/u tail. All evaluation points are held out.
    solver = ConcentricDisk(n_disk=32, n_limb=16)
    table = BasisTable.build(
        solver,
        np.geomspace(0.02, 2, 48),
        np.geomspace(3, 1e8, 201),
        astrometry=True,
    )
    rho = 0.073
    positions = rho * jnp.array([10.37, 37.21, 137.7, 1.374e4, 2.13e7])
    flux, shift = table.joint(positions, rho, a1=0.6)
    expected_flux, expected_shift = solver.joint(positions, rho, a1=0.6)
    np.testing.assert_allclose(flux, expected_flux, rtol=3e-6)
    np.testing.assert_allclose(shift, expected_shift, rtol=3e-5, atol=0)
    assert np.all(np.asarray(shift) > 0)
    np.testing.assert_allclose(positions[-1] * shift[-1], 1.0, rtol=3e-5)


@pytest.fixture(scope="module")
def small_radius_wing_table():
    return BasisTable.build(
        ConcentricDisk(n_disk=64, n_limb=32),
        np.geomspace(1e-6, 1e-3, 96),
        np.geomspace(100, 1e8, 201),
        astrometry=True,
    )


@pytest.mark.parametrize("rho", [1e-6, 1e-5, 1e-3])
@pytest.mark.parametrize("u", [1.0, 100.0])
def test_small_source_fixed_separation_gradients(small_radius_wing_table, u, rho):
    """Unresolved sources must not acquire large interpolation derivatives.

    At fixed u the leading source-average correction is O(rho**2). These
    independent analytic second-moment coefficients have relative neglected
    terms below O(1e-6) for this grid, including both closed radius endpoints.
    """
    coefficient = 0.43
    gamma = 2 * coefficient / (3 - coefficient)
    moment2 = 0.5 - 0.1 * gamma
    expected_radius_response = np.array(
        [
            16 * moment2 * (u * u + 1) / (u**3 * (u * u + 4) ** 2.5),
            -8 * moment2 * (2 * u + 3 / u) / ((u * u + 2) ** 2 * (u * u + 4)),
        ]
    )

    def observable(position, radius):
        return jnp.stack(small_radius_wing_table.joint(position, radius, a1=coefficient))

    du, drho = jax.jit(jax.jacfwd(observable, argnums=(0, 1)))(u, rho)
    if u == 1.0:
        # The resolved finite-source response must retain its physical slope.
        # At the smallest rho, O(rho**2) value corrections approach float64
        # quantization and their interpolated derivatives have an absolute
        # roundoff floor. Keep it explicit rather than asserting unrealizable
        # relative derivative accuracy on these nearly point-source values.
        for index, tolerance in enumerate((1e-8, 5e-9)):
            np.testing.assert_allclose(
                drho[index], rho * expected_radius_response[index], rtol=3e-3, atol=tolerance
            )
    else:
        # At u=100 the value correction can be below float64 resolution.
        # Bound absolute derivative noise instead of asserting relative
        # accuracy on that unrepresentable signal. The corresponding
        # d/dlog(rho) error is at most 1e-11 for the full tested radius range.
        error = np.abs(drho - rho * expected_radius_response)
        assert error[0] < 1e-8
        assert error[1] < 1e-9
    point_du = np.array([-8 / (u**2 * (u * u + 4) ** 1.5), (2 - u * u) / (u * u + 2) ** 2])
    np.testing.assert_allclose(du, point_du, rtol=2e-5, atol=1e-13)


def test_point_limit_at_huge_position_has_finite_derivatives(polynomial_joint_table):
    def observable(theta):
        return jnp.stack(polynomial_joint_table.joint(theta[0], theta[1], a1=theta[2]))

    theta = jnp.array([1e200, 0.0, 0.43])
    forward = jax.jit(jax.jacfwd(observable))(theta)
    reverse = jax.jit(jax.jacrev(observable))(theta)
    assert np.isfinite(forward).all()
    assert np.isfinite(reverse).all()
    reference = jax.jacfwd(lambda u: jnp.stack((pspl(u), point_source_shift(u))))(theta[0])
    np.testing.assert_allclose(forward[:, 0], reference, rtol=2e-12, atol=0)
    np.testing.assert_allclose(forward[:, 1:], 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(reverse, forward, rtol=2e-12, atol=0)


def test_photometric_tables_remain_usable_and_refuse_missing_moments(tmp_path):
    table = BasisTable.build(
        PolynomialJointBases(), np.geomspace(0.01, 1, 6), np.linspace(0, 5, 16)
    )
    assert not table.has_astrometry
    for method in (table.joint_basis, table.joint, table.shift):
        with pytest.raises(ValueError, match="astrometr|moment"):
            method(0.1, 0.1)
    # Old version-one files must keep their original raw-z spline convention.
    path = tmp_path / "version-one.npz"
    np.savez(
        path,
        format_version=1,
        rho_grid=table.rho_grid,
        z_grid=table.z_grid,
        knots_rho=np.asarray(table._knots_rho),
        knots_z=np.asarray(table._knots_z),
        coefficients=np.asarray(table._coefficients),
        a1=table.a1,
    )
    restored = BasisTable.load(path)
    assert not restored.has_astrometry
    np.testing.assert_array_equal(
        restored.A(jnp.array([0, 0.01, 0.1]), 0.1), table.A(jnp.array([0, 0.01, 0.1]), 0.1)
    )


def test_legacy_raw_moment_table_keeps_its_encoding(tmp_path):
    from scipy.interpolate import RectBivariateSpline

    radii, ratios = np.geomspace(0.01, 1, 8), np.linspace(0, 5, 16)
    x, y = np.meshgrid(np.log(radii), np.log1p(ratios**2), indexing="ij")

    def channels(x, y):
        return 0.3 * x + 0.02 * y**2, -0.2 * x + 0.03 * y, -0.2 + 0.03 * x, 0.05 + 0.04 * y

    splines = [
        RectBivariateSpline(np.log(radii), np.log1p(ratios**2), value, s=0)
        for value in channels(x, y)
    ]
    tx, ty = splines[0].get_knots()
    coefficients = np.stack(
        [s.get_coeffs().reshape(len(radii), len(ratios)) for s in splines], axis=-1
    )
    path = tmp_path / "version-two-raw-moments.npz"
    np.savez(
        path,
        format_version=2,
        rho_grid=radii,
        z_grid=ratios,
        knots_rho=tx,
        knots_z=ty,
        coefficients=coefficients,
        a1=0.5,
        coordinate="log1p_ratio2",
        metadata="{}",
    )
    table = BasisTable.load(path)
    assert table.has_astrometry and table.representation == "raw"
    rho, u = 0.073, jnp.array([0.0, 0.017, 0.11, 0.21])
    raw = channels(np.log(rho), np.log1p((u / rho) ** 2))
    expected = (np.exp(raw[0]), np.exp(raw[1]), np.full(u.shape, raw[2]), raw[3])
    for actual, value in zip(table.joint_basis(u, rho), expected):
        np.testing.assert_allclose(actual, value, rtol=3e-13, atol=3e-14)
