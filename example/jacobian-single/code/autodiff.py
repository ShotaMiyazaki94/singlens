"""Annual-parallax light curves and derivatives with respect to event parameters."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import singlens as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from example._common import save_figure, style

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"


def comparison_data(fft=None):
    """Uniform/linear values and raw derivatives, at identical event geometry."""
    times = 8000.0 + jnp.linspace(-40.0, 40.0, 1001)
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)
    base_theta = jnp.array([8000.0, 0.05, 80.0, 0.1, 0.2, -0.1])
    results = {}
    for source in ("uniform", "linear"):
        theta = (
            jnp.concatenate((base_theta, jnp.array([0.5]))) if source == "linear" else base_theta
        )

        def evaluate(theta):
            return sl.magnification(
                times,
                *theta[:3],
                rho=theta[3],
                a1=theta[-1] if source == "linear" else None,
                parallax=geometry,
                pi_e_n=theta[4],
                pi_e_e=theta[5],
                fft=fft,
            )

        curve = jax.jit(evaluate)(theta)
        derivatives = jax.jit(jax.jacfwd(evaluate))(theta)
        jacobian = sl.Parameters(
            t0=derivatives[:, 0],
            u0=derivatives[:, 1],
            t_e=derivatives[:, 2],
            rho=derivatives[:, 3],
            a1=derivatives[:, -1] if source == "linear" else None,
            pi_e_n=derivatives[:, 4],
            pi_e_e=derivatives[:, 5],
        )
        results[source] = (curve, jacobian)
    return times, results


def sensitivity_figure(fft=None):
    """Magnification and raw partial derivatives with identical time limits."""
    fft = sl.FFTConfig() if fft is None else fft
    times, results = comparison_data(fft)
    plot_times = times - 8000.0
    fields = ("t0", "t_e", "u0", "rho", "a1")
    labels = (
        r"$\partial A/\partial t_0$",
        r"$\partial A/\partial t_E$",
        r"$\partial A/\partial u_0$",
        r"$\partial A/\partial\rho$",
        r"$\partial A/\partial a_1$",
    )
    fields += ("pi_e_n", "pi_e_e")
    labels += (r"$\partial A/\partial\pi_{E,N}$", r"$\partial A/\partial\pi_{E,E}$")
    fig, axes = plt.subplots(
        1 + len(fields),
        1,
        figsize=(10, 15),
        gridspec_kw={"height_ratios": [2.4] + [1] * len(fields)},
        sharex=True,
        layout="constrained",
    )
    for source, label in [("uniform", "Uniform"), ("linear", "Linear (a₁ = 0.5)")]:
        curve, jacobian = results[source]
        (line,) = axes[0].plot(plot_times, curve, label=label, lw=1.6)
        for ax, field in zip(axes[1:], fields):
            derivative = getattr(jacobian, field)
            if derivative is not None:
                ax.plot(plot_times, derivative, color=line.get_color(), lw=1.1)
    axes[0].set_ylabel("Magnification A(t)")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="outside upper right",
        ncol=2,
        fontsize=11,
    )
    for ax, label in zip(axes[1:], labels):
        ax.set_ylabel(label)
        ax.axhline(0, color="0.7", lw=0.6, zorder=0)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
    axes[-1].set_xlim(float(plot_times[0]), float(plot_times[-1]))
    fig.align_ylabels(axes)
    fig.supxlabel("HJD − 2458000 [day]")
    fig.suptitle("Annual parallax", x=0.01, ha="left")
    return fig


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    style()
    save_figure(sensitivity_figure(), "jacobian", args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
