"""Check installed resources and event evaluation independently of the checkout."""

from pathlib import Path

import jax
import numpy as np

import singlens


def main():
    checkout = Path(__file__).resolve().parents[1]
    installed = Path(singlens.__file__).resolve()
    if checkout / "src" / "singlens" in installed.parents:
        raise RuntimeError(f"Expected an installed wheel, imported checkout: {installed}")
    jax.config.update("jax_enable_x64", True)
    geometry = singlens.AnnualParallax(t_ref=8000.0, ra_deg=266.4, dec_deg=-29.0)
    model = singlens.Model(source="linear", parallax=geometry)
    params = singlens.Parameters(
        t0=8000.0, u0=0.01, t_e=20.0, rho=0.01, a1=0.5, pi_e_n=0.1, pi_e_e=0.1
    )
    value = jax.jit(model.magnification)(8000.0, params)
    shift = jax.jit(model.astrometric_shift)(8000.0, params)
    slope = jax.grad(lambda t: model.magnification(t, params))(8000.0)
    assert np.isfinite(value) and value > 1
    assert np.isfinite(shift) and np.isfinite(slope)
    print(f"Installed wheel OK: {installed}, JAX={jax.__version__}")


if __name__ == "__main__":
    main()
