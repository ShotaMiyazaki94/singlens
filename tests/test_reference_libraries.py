"""Optional scientific cross-checks; the core suite does not require these libraries."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from singlens import AnnualParallax, LinearLimbDarkened, Model, Parameters, UniformDisk

pytestmark = pytest.mark.reference


@pytest.mark.parametrize("a1", [None, 0.2, 0.5, 0.8])
def test_vbmicrolensing(a1):
    vbml = pytest.importorskip("VBMicrolensing")
    reference = vbml.VBMicrolensing()
    reference.Tol = 1e-6
    reference.a1 = 0.0 if a1 is None else a1
    source = UniformDisk() if a1 is None else LinearLimbDarkened(a1)
    evaluate = jax.jit(source.A)
    for rho in [1e-4, 1e-3, 1e-2, 0.1, 1.0]:
        u = rho * np.array([0.1, 0.5, 0.9, 1.0, 1.1, 2.0, 5.0])
        # ESPLMagDark always evaluates the finite-source LD integral;
        # ESPLMag2 is VBMicrolensing's uniform-source interface.
        f = reference.ESPLMag2 if a1 is None else reference.ESPLMagDark
        expected = [f(float(x), rho) for x in u]
        np.testing.assert_allclose(evaluate(u, rho), expected, rtol=5e-3)


def test_mulensmodel_parallax():
    mm = pytest.importorskip("MulensModel")
    import astropy.units as units
    from astropy.coordinates import SkyCoord

    coords = "17:45:40 -29:00:28"
    c = SkyCoord(coords, frame="icrs", unit=(units.hourangle, units.deg))
    t = 8000.0 + np.linspace(-150.0, 150.0, 200)
    params = {
        "t_0": 2458000.0,
        "t_0_par": 2458000.0,
        "u_0": 0.01,
        "t_E": 30.0,
        "pi_E_N": 0.7,
        "pi_E_E": -0.7,
    }
    ref = mm.Model(params, coords=coords)
    ref.parallax(earth_orbital=True, satellite=False)
    trajectory = ref.get_trajectory(times=t + 2450000.0)
    geometry = AnnualParallax(t_ref=8000.0, ra_deg=c.ra.deg, dec_deg=c.dec.deg)
    model = Model(parallax=geometry)
    params = Parameters(t0=8000.0, u0=0.01, t_e=30.0, pi_e_n=0.7, pi_e_e=-0.7)
    w = model.trajectory(jnp.array(t), params)
    residual = abs(np.asarray(w) - (trajectory.x + 1j * trajectory.y))
    assert residual.max() < 1e-3
    assert residual.mean() < 3e-4
