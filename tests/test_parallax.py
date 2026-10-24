import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens import AnnualParallax, Model, Parameters
from singlens.trajectory import (
    HeliocentricEphemeris,
    compute_parallax,
    load_builtin_earth_ephemeris,
    set_parallax,
)
from singlens.trajectory.ephemeris import interp_uniform_linear, load_horizons_vectors_file


@pytest.fixture(scope="module")
def eph():
    return load_builtin_earth_ephemeris()


def test_bundled_data(eph):
    assert eph.r.shape == eph.v.shape == (len(eph.t), 3)
    np.testing.assert_allclose(np.diff(eph.t), 1.0)
    assert float(eph.t[0]) == 2451544.5
    assert np.isfinite(eph.r).all()


@pytest.mark.parametrize("ephemeris", [False, True])
def test_reference_epoch_zero_jit_and_shapes(eph, ephemeris):
    tref = 8000.0
    setup = (
        AnnualParallax(t_ref=tref, ra_deg=266.0, dec_deg=-29.0, ephemeris=eph)
        if ephemeris
        else set_parallax(tref, 0.0, 0.0, 266.0, -29.0)
    )

    def compute(t, north, east, setup):
        if ephemeris:
            return setup.offsets(t, pi_e_n=north, pi_e_e=east)
        return compute_parallax(t, north, east, setup)

    np.testing.assert_allclose(compute(tref, 0.2, -0.1, setup), 0.0, atol=1e-12)
    times = jnp.array([7950.0, 8000.0, 8050.0])
    f = jax.jit(lambda p: jnp.stack(compute(times, p[0], p[1], setup)))
    p = jnp.array([0.2, -0.1])
    assert np.isfinite(jax.jacrev(f)(p)).all()
    np.testing.assert_allclose(f(p) * 2, f(p * 2), atol=1e-13)
    np.testing.assert_allclose(f(jnp.zeros(2)), 0.0, atol=1e-15)
    if ephemeris:
        model = Model(source="point", parallax=setup)
        params = Parameters(t0=tref, u0=0.1, t_e=30.0, pi_e_n=0.2, pi_e_e=-0.1)
        assert model.trajectory(times.reshape(1, 3), params).shape == (1, 3)
        assert model.magnification(tref, params).shape == ()


def test_ephemeris_out_of_range_and_invalid_tables(eph, tmp_path):
    assert np.isnan(
        interp_uniform_linear(jnp.array([-1.0, 3.0]), 0.0, 1.0, jnp.array([1.0, 2.0, 3.0]))
    ).all()
    for times in [[1.0, 1.0], [1.0, 2.0, 4.0]]:
        tab = np.zeros((len(times), 7))
        tab[:, 0] = times
        with pytest.raises(ValueError, match="uniformly"):
            HeliocentricEphemeris.from_horizons_vectors_table(tab)
    file = tmp_path / "empty.txt"
    file.write_text("not an ephemeris")
    with pytest.raises(ValueError, match="No ephemeris"):
        load_horizons_vectors_file(file)


@pytest.mark.parametrize("use_hjd", [False, True])
def test_absolute_time_equivalence(eph, use_hjd):
    relative = AnnualParallax(
        t_ref=8000.0, ra_deg=266.0, dec_deg=-29.0, ephemeris=eph, use_hjd=use_hjd
    )
    absolute = AnnualParallax(
        t_ref=2458000.0,
        ra_deg=266.0,
        dec_deg=-29.0,
        ephemeris=eph,
        use_hjd=use_hjd,
        time_origin=0.0,
    )
    times = jnp.array([7950.0, 8000.0, 8050.0])
    np.testing.assert_allclose(
        relative.offsets(times, pi_e_n=0.1, pi_e_e=0.2),
        absolute.offsets(times + 2450000.0, pi_e_n=0.1, pi_e_e=0.2),
    )
