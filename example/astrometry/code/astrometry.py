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
from example._common import COLORS, save_figure, save_json, style

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"


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
    event = dict(t0=0.0, u0=0.03, t_e=20.0)
    rho = 0.1
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
    )


def _zero_lines(ax):
    ax.axhline(0, color="0.65", linewidth=0.8, zorder=0)
    ax.axvline(0, color="0.65", linewidth=0.8, zorder=0)


def _contact_lines(ax, data):
    contact = data["event"]["t_e"] * np.sqrt(data["rho"] ** 2 - data["event"]["u0"] ** 2)
    for time in (-contact, contact):
        ax.axvline(time, color="0.6", linestyle=":", linewidth=1, zorder=0)


def _time_direction(ax, shift, color):
    # Same physical epochs in all tracks: +10.0 to +11.0 days.
    ax.annotate(
        "",
        xy=(shift[2550].real, shift[2550].imag),
        xytext=(shift[2500].real, shift[2500].imag),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )


def source_profiles_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax_flux, ax_tau, ax_beta, ax_track = axes.flat
    labels = ["Point source", "Uniform disk", r"Linear ($a_1=0.6$)"]
    line_styles = ["--", "-", "-"]
    times = data["times"]
    for label, color, ls, A, shift in zip(
        labels, COLORS, line_styles, data["magnifications"], data["shifts"]
    ):
        ax_flux.plot(times, A, label=label, color=color, ls=ls, lw=1.8)
        ax_tau.plot(times, shift.real, color=color, ls=ls, lw=1.8)
        ax_beta.plot(times, shift.imag, color=color, ls=ls, lw=1.8)
        ax_track.plot(shift.real, shift.imag, color=color, ls=ls, lw=1.8)
        _time_direction(ax_track, shift, color)
    for ax in (ax_flux, ax_tau, ax_beta):
        ax.set(xlim=(-6, 6), xlabel=r"$t-t_0$ [day]")
        _contact_lines(ax, data)
    ax_flux.set(title="Light curve · source crossing", ylabel="Magnification A")
    ax_flux.legend(fontsize=10, loc="upper left")
    ax_tau.set(
        title="Shift along the trajectory", ylabel=r"$\Delta\theta_\tau$ [mas]", ylim=(-0.19, 0.19)
    )
    ax_beta.set(title="Shift perpendicular to the trajectory", ylabel=r"$\Delta\theta_\beta$ [mas]")
    ax_tau.axhline(0, color="0.65", linewidth=0.8, zorder=0)
    ax_beta.axhline(0, color="0.65", linewidth=0.8, zorder=0)
    ax_track.set(
        title="Centroid-shift track · −40 to +40 day",
        xlabel=r"$\Delta\theta_\tau$ [mas]",
        ylabel=r"$\Delta\theta_\beta$ [mas]",
    )
    _zero_lines(ax_track)
    ax_track.text(
        0.03,
        0.08,
        "Arrows: increasing time\nAxis scales differ",
        transform=ax_track.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="0.3",
    )
    inset = ax_track.inset_axes([0.56, 0.10, 0.41, 0.43])
    for color, ls, shift in zip(COLORS, line_styles, data["shifts"]):
        inset.plot(shift.real, shift.imag, color=color, ls=ls, lw=1.4)
    inset.set(xlim=(-0.055, 0.055), ylim=(-0.025, 0.025), title="Near closest approach")
    inset.title.set_fontsize(9)
    inset.tick_params(labelsize=8)
    _zero_lines(inset)
    inset.set_facecolor("#FFFFFF")
    fig.suptitle(
        r"Finite-source photometry and astrometry  |  $u_0=0.03$, $\rho=0.1$, "
        r"$t_E=20$ day, $\theta_E=1.2$ mas",
        fontsize=14,
    )
    fig.supxlabel(
        "Dotted vertical lines: lens on the source limb. Shifts are relative to the unlensed source.",
        fontsize=10,
    )
    return fig


def blending_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    ax_flux, ax_tau, ax_beta, ax_track = axes.flat
    times = data["times"]
    source_shift = data["shifts"][2]
    lens_shift = np.asarray(data["luminous_lens"].relative_shift)
    field_shift = np.asarray(data["field_blend"].relative_shift)
    labels = ["No blend", "Luminous lens at (0, 0) mas", "Field blend at (3, 2) mas"]
    colors = [COLORS[2], COLORS[1], COLORS[3]]
    line_styles = ["--", "-", "-"]
    ax_flux.plot(
        times, data["magnifications"][2], color=colors[0], ls="--", lw=1.8, label="No blend"
    )
    ax_flux.plot(times, data["normalized_flux"], color=colors[1], lw=3, label="Luminous lens")
    field_normalized_flux = np.asarray(data["field_blend"].total_flux) / (
        data["source_flux"] + data["blend_flux"]
    )
    ax_flux.plot(
        times,
        field_normalized_flux,
        color=colors[2],
        ls="--",
        lw=1.5,
        label="Field blend (same curve)",
    )
    ax_flux.set(
        title="Same blend flux → same light curve",
        xlim=(-6, 6),
        xlabel=r"$t-t_0$ [day]",
        ylabel=r"$F(t)/(F_s+F_b)$",
    )
    _contact_lines(ax_flux, data)
    ax_flux.legend(fontsize=10)
    for label, color, ls, shift in zip(
        labels, colors, line_styles, (source_shift, lens_shift, field_shift)
    ):
        ax_tau.plot(times, shift.real, color=color, ls=ls, lw=1.8)
        ax_beta.plot(times, shift.imag, label=label, color=color, ls=ls, lw=1.8)
        ax_track.plot(shift.real, shift.imag, color=color, ls=ls, lw=1.8)
        _time_direction(ax_track, shift, color)
    for ax in (ax_tau, ax_beta):
        ax.set(xlim=(-40, 40), xlabel=r"$t-t_0$ [day]")
        _zero_lines(ax)
    ax_tau.set(
        title="Shift along the trajectory", ylabel=r"$\Delta\theta_{\mathrm{obs},\tau}$ [mas]"
    )
    ax_beta.set(
        title="Shift perpendicular to the trajectory",
        ylabel=r"$\Delta\theta_{\mathrm{obs},\beta}$ [mas]",
    )
    ax_track.set(
        title="Centroid-shift tracks · −40 to +40 day",
        xlabel=r"$\Delta\theta_{\mathrm{obs},\tau}$ [mas]",
        ylabel=r"$\Delta\theta_{\mathrm{obs},\beta}$ [mas]",
    )
    _zero_lines(ax_track)
    ax_track.set_aspect("equal", adjustable="datalim")
    ax_track.text(
        0.03,
        0.04,
        "Arrows: increasing time",
        transform=ax_track.transAxes,
        fontsize=10,
        color="0.3",
    )
    ax_beta.legend(loc="lower right", fontsize=9)
    fig.suptitle(
        r"Unresolved blending  |  $F_b/F_s=0.5$, linear limb darkening ($a_1=0.6$)",
        fontsize=14,
    )
    fig.supxlabel(
        "Each shift is relative to its own same-time, unlensed combined-light centroid.",
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
    style()
    data = calculate_examples()
    verify_observables(data)
    save_figure(source_profiles_figure(data), "source_profiles", args.output_dir)
    save_figure(blending_figure(data), "blending", args.output_dir)
    save_data(data, args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
