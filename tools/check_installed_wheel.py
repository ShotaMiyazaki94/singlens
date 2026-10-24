"""Smoke-test installed resources and autodiff outside the source package."""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import singlens


def main():
    checkout = Path(__file__).resolve().parents[1]
    installed = Path(singlens.__file__).resolve()
    if checkout / "src" / "singlens" in installed.parents:
        raise RuntimeError(f"Expected an installed wheel, imported checkout: {installed}")
    assert importlib.util.find_spec("singlens.lightcurve") is None
    assert not any(
        hasattr(singlens, name)
        for name in ("magnification", "astrometric_shift", "FFTConfig", "BasisTable")
    )
    for x64 in (False, True):
        jax.config.update("jax_enable_x64", x64)
        geometry = singlens.AnnualParallax(t_ref=8000.0, ra_deg=266.4, dec_deg=-29.0)
        model = singlens.Model(source="linear", parallax=geometry)
        if not x64:
            # Check after resources load, before JAX may initialize LAPACK.
            assert not any(name == "scipy" or name.startswith("scipy.") for name in sys.modules)
        assert model.solver.representation == "point_residual"
        assert model.solver.has_astrometry
        params = singlens.Parameters(
            t0=8000.0,
            u0=0.01,
            t_e=20.0,
            rho=0.01,
            a1=0.5,
            pi_e_n=0.1,
            pi_e_e=0.1,
        )
        value, shift = jax.jit(model.joint)(8000.0, params)
        assert np.isfinite(value) and value > 1
        assert np.isfinite(shift)
        slope = jax.grad(lambda t: model.magnification(t, params))(8000.0)
        assert np.isfinite(slope)
        times = jnp.array([7999.0, 8000.0, 8001.0])
        flux = model.magnification(times, params)
        marginalized = model.marginalize_flux(
            times,
            params,
            flux,
            jnp.ones_like(times) * 0.1,
            prior_mean=jnp.array([1.0, 0.0]),
            prior_covariance=jnp.eye(2),
        )
        assert np.isfinite(marginalized.log_likelihood)
        print(
            f"Installed wheel OK: x64={x64}, Python={sys.version.split()[0]}, JAX={jax.__version__}"
        )


if __name__ == "__main__":
    main()
