"""Finite-source light curves, astrometric shifts, and unresolved blending."""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import singlens as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from example._common import save_figure, save_json

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
COLORS = ("#222222", "#0072B2", "#D55E00")
LINE_STYLES = ("--", "-", "-.")
SOURCE_LABELS = ("Point source", "Uniform disk", r"Linear ($a_1=0.6$)")
BLEND_LABELS = ("No blend", "Luminous lens", "Unrelated blend")


def publication_style():
    """Style this example locally, leaving the other examples unchanged."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
        }
    )


# BEGIN BLENDING MODEL
class BlendedObservables(NamedTuple):
    total_flux: jax.Array
    observed_centroid: jax.Array
    baseline_centroid: jax.Array
    relative_shift: jax.Array


def blended_observables(
    magnification,
    source_shift,
    source_position,
    *,
    source_flux,
    blend_flux,
    blend_position,
):
    """Flux and centroid of one unresolved source plus unmicrolensed light.

    Positions and source_shift are complex coordinates in the same frame and
    angular units. Arrays broadcast over epochs. The caller supplies physical
    scalar source_flux > 0 and blend_flux >= 0; no validation is performed.
    The baseline is the same-time unlensed combined-light centroid, not a
    fixed reference-epoch position. Several unmicrolensed blends can be
    represented by their total flux and flux-weighted position at each epoch.
    """
    A = jnp.asarray(magnification)
    delta = jnp.asarray(source_shift)
    theta_s = jnp.asarray(source_position)
    theta_b = jnp.asarray(blend_position)
    g = jnp.asarray(blend_flux) / jnp.asarray(source_flux)
    total_flux = source_flux * (A + g)
    baseline_centroid = (theta_s + g * theta_b) / (1 + g)
    # Compute the small displacement directly, avoiding subtraction of two
    # potentially large absolute centroids.
    relative_shift = A / (A + g) * delta + g / (1 + g) * (A - 1) / (A + g) * (theta_s - theta_b)
    observed_centroid = baseline_centroid + relative_shift
    return BlendedObservables(total_flux, observed_centroid, baseline_centroid, relative_shift)


# END BLENDING MODEL


def calculate_examples():
    # BEGIN SOURCE EXAMPLE
    times = jnp.linspace(-40.0, 40.0, 4001)  # days
    event = dict(t0=0.0, u0=0.1, t_e=20.0)
    rho = 0.5
    theta_e_mas = 1.2

    point_model = sl.Model(source="point")
    uniform_model = sl.Model(source="uniform")
    linear_model = sl.Model(source="linear")
    point_params = sl.Parameters(**event)
    uniform_params = sl.Parameters(**event, rho=rho)
    linear_params = sl.Parameters(**event, rho=rho, a1=0.6)

    A_point = point_model.magnification(times, point_params)
    A_uniform = uniform_model.magnification(times, uniform_params)
    A_linear = linear_model.magnification(times, linear_params)
    shift_point_mas = theta_e_mas * point_model.astrometric_shift(times, point_params)
    shift_uniform_mas = theta_e_mas * uniform_model.astrometric_shift(times, uniform_params)
    shift_linear_mas = theta_e_mas * linear_model.astrometric_shift(times, linear_params)
    # Real and imaginary parts are the signed tau and beta components.
    # END SOURCE EXAMPLE

    # BEGIN BLENDING EXAMPLE
    # A fixed lens at the origin; source proper motion follows the trajectory.
    source_position_mas = theta_e_mas * linear_model.trajectory(times, linear_params)
    source_flux, blend_flux = 1.0, 0.5
    lens_position_mas = 0.0 + 0.0j
    field_position_mas = 3.0 + 2.0j

    luminous_lens = blended_observables(
        A_linear,
        shift_linear_mas,
        source_position_mas,
        source_flux=source_flux,
        blend_flux=blend_flux,
        blend_position=lens_position_mas,
    )
    field_blend = blended_observables(
        A_linear,
        shift_linear_mas,
        source_position_mas,
        source_flux=source_flux,
        blend_flux=blend_flux,
        blend_position=field_position_mas,
    )
    normalized_flux = luminous_lens.total_flux / (source_flux + blend_flux)
    # field_blend has exactly the same light curve, but a different centroid.
    # END BLENDING EXAMPLE

    # Track plots include logarithmic wings so all curves visibly approach the
    # unlensed origin. No artificial segment closes the finite-time endpoints.
    wings = event["t_e"] * np.geomspace(2.0, 1000.0, 1001)
    track_times = jnp.asarray(np.unique(np.concatenate((-wings, times, wings))))
    track_shifts = [
        np.asarray(theta_e_mas * model.astrometric_shift(track_times, params))
        for model, params in (
            (point_model, point_params),
            (uniform_model, uniform_params),
            (linear_model, linear_params),
        )
    ]
    track_A = linear_model.magnification(track_times, linear_params)
    track_source = theta_e_mas * linear_model.trajectory(track_times, linear_params)
    track_blends = [
        blended_observables(
            track_A,
            track_shifts[2],
            track_source,
            source_flux=source_flux,
            blend_flux=blend_flux,
            blend_position=position,
        )
        for position in (lens_position_mas, field_position_mas)
    ]

    return dict(
        times=np.asarray(times),
        event=event,
        rho=rho,
        theta_e_mas=theta_e_mas,
        a1=float(linear_params.a1),
        magnifications=[np.asarray(A) for A in (A_point, A_uniform, A_linear)],
        shifts=[np.asarray(s) for s in (shift_point_mas, shift_uniform_mas, shift_linear_mas)],
        source_position_mas=np.asarray(source_position_mas),
        source_flux=source_flux,
        blend_flux=blend_flux,
        lens_position_mas=lens_position_mas,
        field_position_mas=field_position_mas,
        luminous_lens=luminous_lens,
        field_blend=field_blend,
        normalized_flux=np.asarray(normalized_flux),
        track_times=np.asarray(track_times),
        track_shifts=track_shifts,
        track_magnification=np.asarray(track_A),
        track_blends=track_blends,
    )


def _contact_lines(ax, data):
    contact = data["event"]["t_e"] * np.sqrt(data["rho"] ** 2 - data["event"]["u0"] ** 2)
    for time in (-contact, contact):
        ax.axvline(time, color="0.65", linestyle=":", linewidth=0.8, zorder=0)


def _time_panels(data):
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.5), sharex=True)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.105, top=0.92, hspace=0)
    for ax, label in zip(axes, ("(a)", "(b)", "(c)")):
        ax.set_xlim(-40, 40)
        ax.minorticks_on()
        ax.text(0.025, 0.88, label, transform=ax.transAxes, fontsize=10)
        _contact_lines(ax, data)
    axes[-1].set_xlabel(r"$t-t_0$ (day)")
    axes[-1].set_xticks(np.arange(-40, 41, 10))
    for ax in axes[1:]:
        ax.axhline(0, color="0.8", linewidth=0.6, zorder=0)
    return fig, axes


def _top_legend(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.56, 0.994),
        ncol=ncol,
        handlelength=2.5,
        columnspacing=1.3,
    )


def _track_arrow(ax, data, shift, color, start, end):
    i = np.argmin(np.abs(data["track_times"] - start))
    j = np.argmin(np.abs(data["track_times"] - end))
    ax.annotate(
        "",
        xy=(1000 * shift[j].real, 1000 * shift[j].imag),
        xytext=(1000 * shift[i].real, 1000 * shift[i].imag),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.0, mutation_scale=10),
    )


def source_profiles_figure(data):
    fig, (ax_flux, ax_tau, ax_beta) = _time_panels(data)
    for label, color, ls, A, shift in zip(
        SOURCE_LABELS, COLORS, LINE_STYLES, data["magnifications"], data["shifts"]
    ):
        ax_flux.plot(data["times"], A, label=label, color=color, ls=ls)
        ax_tau.plot(data["times"], 1000 * shift.real, color=color, ls=ls)
        ax_beta.plot(data["times"], 1000 * shift.imag, color=color, ls=ls)
    ax_flux.set_ylabel(r"Magnification $A$")
    ax_flux.set_ylim(0.8, 10.8)
    ax_flux.text(
        0.97,
        0.88,
        r"$u_0=0.1$, $\rho=0.5$",
        transform=ax_flux.transAxes,
        ha="right",
        va="top",
        fontsize=9,
    )
    ax_tau.set_ylabel(r"$\delta\theta_\tau$ ($\mu$as)")
    ax_tau.set_ylim(-475, 475)
    ax_tau.set_yticks([-400, -200, 0, 200, 400])
    ax_beta.set_ylabel(r"$\delta\theta_\beta$ ($\mu$as)")
    ax_beta.set_ylim(-52, 68)
    ax_beta.set_yticks([-40, -20, 0, 20, 40, 60])
    _top_legend(fig, ax_flux)
    return fig


def source_tracks_figure(data):
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.16, top=0.9)
    for label, color, ls, shift in zip(SOURCE_LABELS, COLORS, LINE_STYLES, data["track_shifts"]):
        ax.plot(1000 * shift.real, 1000 * shift.imag, color=color, ls=ls, label=label)
    _track_arrow(ax, data, data["track_shifts"][0], COLORS[0], -28, -22)
    _track_arrow(ax, data, data["track_shifts"][1], COLORS[1], 3, 4.5)
    _track_arrow(ax, data, data["track_shifts"][2], COLORS[2], 3, 4.5)
    ax.plot(0, 0, marker="+", color="0.15", markersize=6, markeredgewidth=0.8, zorder=5)
    ax.set(
        xlim=(-460, 460),
        ylim=(-36, 66),
        xlabel=r"$\delta\theta_\tau$ ($\mu$as)",
        ylabel=r"$\delta\theta_\beta$ ($\mu$as)",
    )
    # One microarcsecond is drawn five times longer vertically than horizontally.
    # This deliberate display expansion reveals the transit loop; it does not
    # rescale any physical coordinate or data value.
    ax.set_aspect(5)
    ax.set_xticks([-400, -200, 0, 200, 400])
    ax.set_yticks([-20, 0, 20, 40, 60])
    ax.minorticks_on()
    _top_legend(fig, ax)
    return fig


def blending_figure(data):
    fig, (ax_flux, ax_tau, ax_beta) = _time_panels(data)
    ax_flux.plot(
        data["times"], data["magnifications"][2], color=COLORS[0], ls="--", label="No blend"
    )
    # Both blend positions produce precisely the same flux; draw this only once.
    ax_flux.plot(data["times"], data["normalized_flux"], color="0.45", label="Both blends")
    ax_flux.set_ylabel(r"$F/(F_s+F_b)$")
    ax_flux.set_ylim(0.8, 5.1)
    ax_flux.legend(loc="upper right", handlelength=2.5)
    ax_flux.text(0.03, 0.68, r"$F_b/F_s=0.5$", transform=ax_flux.transAxes, fontsize=9)
    shifts = (
        data["shifts"][2],
        data["luminous_lens"].relative_shift,
        data["field_blend"].relative_shift,
    )
    for label, color, ls, shift in zip(BLEND_LABELS, COLORS, LINE_STYLES, shifts):
        shift = 1000 * np.asarray(shift)
        ax_tau.plot(data["times"], shift.real, color=color, ls=ls, label=label)
        ax_beta.plot(data["times"], shift.imag, color=color, ls=ls)
    ax_tau.set_ylabel(r"$\delta\theta_{\mathrm{obs},\tau}$ ($\mu$as)")
    ax_beta.set_ylabel(r"$\delta\theta_{\mathrm{obs},\beta}$ ($\mu$as)")
    # The blend-free beta shift is close to the top of this panel.
    ax_beta.texts[0].set_y(0.72)
    _top_legend(fig, ax_tau)
    return fig


def blending_tracks_figure(data):
    fig, ax = plt.subplots(figsize=(6.4, 3.65))
    fig.subplots_adjust(left=0.15, right=0.97, bottom=0.18, top=0.88)
    shifts = (
        data["track_shifts"][2],
        *(np.asarray(b.relative_shift) for b in data["track_blends"]),
    )
    for label, color, ls, shift in zip(BLEND_LABELS, COLORS, LINE_STYLES, shifts):
        ax.plot(1000 * shift.real, 1000 * shift.imag, color=color, ls=ls, label=label)
        _track_arrow(ax, data, shift, color, -10, -8)
    ax.plot(0, 0, marker="+", color="0.15", markersize=6, markeredgewidth=0.8, zorder=5)
    ax.set(
        xlabel=r"$\delta\theta_{\mathrm{obs},\tau}$ ($\mu$as)",
        ylabel=r"$\delta\theta_{\mathrm{obs},\beta}$ ($\mu$as)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.minorticks_on()
    _top_legend(fig, ax)
    return fig


def limb_darkening_figure(data):
    fig, (ax_flux, ax_tau, ax_beta) = _time_panels(data)
    relative_flux = 100 * (data["magnifications"][2] / data["magnifications"][1] - 1)
    shift = 1000 * (data["shifts"][2] - data["shifts"][1])
    for ax, value in zip((ax_flux, ax_tau, ax_beta), (relative_flux, shift.real, shift.imag)):
        ax.plot(data["times"], value, color=COLORS[2])
    ax_flux.axhline(0, color="0.8", linewidth=0.6, zorder=0)
    ax_flux.set_ylabel(r"$\Delta A/A_{\mathrm{uni}}$ (%)")
    ax_tau.set_ylabel(r"$\Delta\delta\theta_\tau$ ($\mu$as)")
    ax_beta.set_ylabel(r"$\Delta\delta\theta_\beta$ ($\mu$as)")
    ax_flux.set_yticks([-4, 0, 4, 8])
    ax_tau.set_yticks([-30, 0, 30])
    ax_beta.set_yticks([0, 2, 4, 6])
    fig.text(
        0.56,
        0.975,
        r"Linear limb darkening ($a_1=0.6$) $-$ uniform disk",
        ha="center",
        va="top",
        fontsize=10,
    )
    return fig


def verify_observables(data):
    """Check the flux-weighted identity and physical limiting cases."""
    A = np.asarray(data["magnifications"][2])
    delta = np.asarray(data["shifts"][2])
    theta_s = np.asarray(data["source_position_mas"])
    fs, fb = data["source_flux"], data["blend_flux"]
    for name, theta_b in (
        ("luminous_lens", data["lens_position_mas"]),
        ("field_blend", data["field_position_mas"]),
    ):
        result = data[name]
        direct = (fs * A * (theta_s + delta) + fb * theta_b) / (fs * A + fb)
        np.testing.assert_allclose(result.observed_centroid, direct, rtol=2e-13, atol=2e-14)
        np.testing.assert_allclose(
            result.relative_shift, direct - result.baseline_centroid, rtol=2e-11, atol=2e-14
        )
    no_blend = blended_observables(
        A, delta, theta_s, source_flux=fs, blend_flux=0.0, blend_position=3 + 2j
    )
    np.testing.assert_allclose(no_blend.relative_shift, delta, atol=0, rtol=0)
    unlensed = blended_observables(
        jnp.ones_like(A),
        jnp.zeros_like(delta),
        theta_s,
        source_flux=fs,
        blend_flux=fb,
        blend_position=3 + 2j,
    )
    np.testing.assert_allclose(unlensed.relative_shift, 0.0, atol=0, rtol=0)
    coincident = blended_observables(
        A, delta, theta_s, source_flux=fs, blend_flux=fb, blend_position=theta_s
    )
    np.testing.assert_allclose(
        coincident.relative_shift, A / (A + fb / fs) * delta, rtol=2e-13, atol=2e-14
    )
    np.testing.assert_array_equal(data["luminous_lens"].total_flux, data["field_blend"].total_flux)
    assert (
        np.max(
            np.abs(
                np.asarray(
                    data["luminous_lens"].relative_shift - data["field_blend"].relative_shift
                )
            )
        )
        > 0.1
    )
    # The recipe is usable inside the same JAX transformations as singlens.
    compiled = jax.jit(blended_observables)(
        A, delta, theta_s, source_flux=fs, blend_flux=fb, blend_position=3 + 2j
    )
    np.testing.assert_allclose(
        compiled.relative_shift, data["field_blend"].relative_shift, rtol=2e-13, atol=2e-14
    )
    blend_gradient = jax.jacfwd(
        lambda flux: (
            blended_observables(
                A, delta, theta_s, source_flux=fs, blend_flux=flux, blend_position=3 + 2j
            ).relative_shift
        )
    )(fb)
    assert np.all(np.isfinite(np.asarray(blend_gradient)))


def save_data(data, output):
    output.mkdir(parents=True, exist_ok=True)
    columns = [data["times"]]
    headers = ["time_day"]
    for name, A, shift in zip(
        ("point", "uniform", "linear"), data["magnifications"], data["shifts"]
    ):
        columns.extend([A, shift.real, shift.imag])
        headers.extend([f"A_{name}", f"delta_tau_{name}_mas", f"delta_beta_{name}_mas"])
    np.savetxt(
        output / "source_profiles.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
    columns = [
        data["times"],
        data["magnifications"][2],
        data["normalized_flux"],
        data["source_position_mas"].real,
        data["source_position_mas"].imag,
    ]
    headers = [
        "time_day",
        "A_linear",
        "normalized_blended_flux",
        "source_tau_mas",
        "source_beta_mas",
    ]
    for name in ("luminous_lens", "field_blend"):
        for field in ("observed_centroid", "baseline_centroid", "relative_shift"):
            value = np.asarray(getattr(data[name], field))
            columns.extend([value.real, value.imag])
            headers.extend([f"{name}_{field}_tau_mas", f"{name}_{field}_beta_mas"])
    np.savetxt(
        output / "blending.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
    columns = [data["track_times"]]
    headers = ["time_day"]
    for name, shift in zip(("point", "uniform", "linear"), data["track_shifts"]):
        columns.extend([shift.real, shift.imag])
        headers.extend([f"delta_tau_{name}_mas", f"delta_beta_{name}_mas"])
    for name, result in zip(("luminous_lens", "field_blend"), data["track_blends"]):
        shift = np.asarray(result.relative_shift)
        columns.extend([shift.real, shift.imag])
        headers.extend([f"delta_tau_{name}_mas", f"delta_beta_{name}_mas"])
    np.savetxt(
        output / "tracks.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
    limb_darkening_shift = data["shifts"][2] - data["shifts"][1]
    np.savetxt(
        output / "limb_darkening.csv",
        np.column_stack(
            [
                data["times"],
                100 * (data["magnifications"][2] / data["magnifications"][1] - 1),
                limb_darkening_shift.real,
                limb_darkening_shift.imag,
            ]
        ),
        delimiter=",",
        header="time_day,relative_flux_percent,delta_tau_linear_minus_uniform_mas,"
        "delta_beta_linear_minus_uniform_mas",
        comments="",
        fmt="%.12g",
    )
    save_json(
        dict(
            event=data["event"],
            rho=data["rho"],
            a1=data["a1"],
            theta_e_mas=data["theta_e_mas"],
            source_flux=data["source_flux"],
            blend_flux=data["blend_flux"],
            lens_position_mas=[data["lens_position_mas"].real, data["lens_position_mas"].imag],
            field_position_mas=[data["field_position_mas"].real, data["field_position_mas"].imag],
            coordinate_frame="tau + 1j*beta; fixed lens at origin",
            shift_reference="same-time unlensed source, or unlensed source-plus-blend centroid",
            samples=len(data["times"]),
            time_range_day=[float(data["times"][0]), float(data["times"][-1])],
            track_samples=len(data["track_times"]),
            track_time_range_day=[float(data["track_times"][0]), float(data["track_times"][-1])],
            track_max_abs_tau=1000.0,
            plot_angular_unit="microarcsecond; CSV data use milliarcseconds",
            source_track_vertical_expansion=5.0,
            blending_track_equal_angular_axes=True,
            jax_x64=bool(jax.config.jax_enable_x64),
        ),
        "configuration",
        output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    publication_style()
    data = calculate_examples()
    verify_observables(data)
    save_figure(source_profiles_figure(data), "source_profiles", args.output_dir)
    save_figure(source_tracks_figure(data), "source_tracks", args.output_dir)
    save_figure(blending_figure(data), "blending", args.output_dir)
    save_figure(blending_tracks_figure(data), "blending_tracks", args.output_dir)
    save_figure(limb_darkening_figure(data), "limb_darkening", args.output_dir)
    save_data(data, args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
