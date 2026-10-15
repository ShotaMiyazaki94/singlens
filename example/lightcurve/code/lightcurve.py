"""Annual-parallax light curves from microlensing parameters."""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import singlens as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from example._common import save_figure, style

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"


def lightcurve_figure():
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    times = 8000.0 + jnp.linspace(-40.0, 40.0, 1001)
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)
    event = dict(t0=8000.0, u0=0.05, t_e=80.0, pi_e_n=0.2, pi_e_e=-0.1)
    for label, source in [
        ("Point", {}),
        ("Uniform", dict(rho=0.1)),
        ("Linear (a₁ = 0.5)", dict(rho=0.1, a1=0.5)),
    ]:
        ax.plot(
            times - 8000.0,
            sl.magnification(times, **event, **source, parallax=geometry),
            label=label,
        )
    ax.set(title="Annual parallax", xlabel="HJD − 2458000 [day]", ylabel="Magnification")
    ax.legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    style()
    save_figure(lightcurve_figure(), "lightcurve", args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
