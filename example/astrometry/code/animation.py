"""Animate the light curve and astrometric trajectory at the same epochs."""

import argparse
import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"


def save_source_animation(data, output):
    """Save a uniformly timed, looping GIF and return its path.

    The 241 epochs cover -4 to +4 Einstein times relative to closest approach
    (-80 to +80 days for this example), at 20 frames per second. The right
    panel expands the vertical display scale by five, retaining physical
    microarcsecond coordinates. Faint curves show the complete trajectories;
    the colored trails cover only the animated time interval.
    """
    # A local import lets astrometry.main call this function without a circular
    # import, and supports running this file both directly and with python -m.
    from example.astrometry.code.astrometry import (
        COLORS,
        LINE_STYLES,
        SOURCE_LABELS,
    )

    event = data["event"]
    contact = event["t_e"] * np.sqrt(data["rho"] ** 2 - event["u0"] ** 2)
    return save_event_animation(
        data["times"] - event["t0"],
        data["magnifications"],
        [1000 * shift for shift in data["shifts"]],
        output,
        labels=SOURCE_LABELS,
        colors=COLORS,
        line_styles=LINE_STYLES,
        filename="source_animation.gif",
        full_shifts=[1000 * shift for shift in data["track_shifts"]],
        contact_times=(-contact, contact),
        vertical_expansion=5,
        flux_ylim=(0.8, 10.8),
        track_xlim=(-460, 460),
        track_ylim=(-36, 66),
        flux_xticks=np.arange(-80, 81, 40),
        flux_yticks=[2, 4, 6, 8, 10],
        track_xticks=[-400, -200, 0, 200, 400],
        track_yticks=[-20, 0, 20, 40, 60],
    )


def save_event_animation(
    times,
    magnifications,
    shifts,
    output,
    *,
    labels,
    colors,
    line_styles,
    filename,
    full_shifts=None,
    contact_times=(),
    vertical_expansion=1,
    flux_ylim=None,
    track_xlim=None,
    track_ylim=None,
    flux_xticks=None,
    flux_yticks=None,
    track_xticks=None,
    track_yticks=None,
    flux_ylabel=r"Magnification $A$",
    track_xlabel=r"$\delta\theta_\tau$ ($\mu$as)",
    track_ylabel=r"$\delta\theta_\beta$ ($\mu$as)",
):
    """Render synchronized light curves and centroid tracks from model arrays.

    ``times`` are relative days on a uniform grid of 240*n + 1 samples (e.g.
    4801), and ``shifts`` are complex microarcseconds. The GIF shows 241
    uniformly spaced epochs at 20 fps. One array per model is supplied in
    ``magnifications`` and ``shifts``; optional ``full_shifts`` supply longer
    faint background trajectories. Angular axes default to equal scales.
    """
    from example.astrometry.code.astrometry import publication_style

    publication_style()
    relative_times = np.asarray(times)
    if len(relative_times) < 241 or (len(relative_times) - 1) % 240:
        raise ValueError("Use a uniform time grid of 240*n + 1 samples, with n >= 1.")
    spacing = np.diff(relative_times)
    if spacing[0] <= 0 or not np.allclose(spacing, spacing[0]):
        raise ValueError("Animation times must increase with uniform spacing.")
    frame_indices = np.linspace(0, len(relative_times) - 1, 241, dtype=int)
    magnifications = [np.asarray(A) for A in magnifications]
    shifts = [np.asarray(shift) for shift in shifts]
    full_shifts = shifts if full_shifts is None else [np.asarray(s) for s in full_shifts]
    if flux_ylim is None:
        flux_values = np.concatenate(magnifications)
        flux_padding = 0.08 * np.ptp(flux_values)
        flux_ylim = (flux_values.min() - flux_padding, flux_values.max() + flux_padding)

    fig = plt.figure(figsize=(10.5, 4.0), dpi=110)
    # Matching panel heights, with room for a shared legend and axis labels.
    panel_height = 5 * (66 + 36) / (460 + 460) * (0.405 * 10.5) / 4.0
    ax_flux = fig.add_axes((0.065, 0.17, 0.405, panel_height))
    ax_track = fig.add_axes((0.575, 0.17, 0.405, panel_height))
    all_shifts = np.concatenate(full_shifts + shifts)
    if track_xlim is None or track_ylim is None:
        x_center = (all_shifts.real.min() + all_shifts.real.max()) / 2
        y_center = (all_shifts.imag.min() + all_shifts.imag.max()) / 2
        x_span = 1.14 * np.ptp(all_shifts.real)
        y_span = 1.14 * np.ptp(all_shifts.imag)
        # Expand bounds to fill the panel while preserving the specified scale.
        span_ratio = panel_height * 4.0 / (0.405 * 10.5 * vertical_expansion)
        x_span = max(x_span, y_span / span_ratio)
        y_span = x_span * span_ratio
        if track_xlim is None:
            track_xlim = (x_center - x_span / 2, x_center + x_span / 2)
        if track_ylim is None:
            track_ylim = (y_center - y_span / 2, y_center + y_span / 2)
    ax_flux.set(
        xlim=(relative_times[0], relative_times[-1]),
        ylim=flux_ylim,
        xlabel=r"$t-t_0$ (day)",
        ylabel=flux_ylabel,
        title="Light curve",
    )
    track_title = "Astrometric shift (equal angular scales)"
    if vertical_expansion != 1:
        track_title = rf"Astrometric shift (vertical scale $\times {vertical_expansion:g}$)"
    ax_track.set(
        xlim=track_xlim,
        ylim=track_ylim,
        xlabel=track_xlabel,
        ylabel=track_ylabel,
        title=track_title,
    )
    ax_track.set_aspect(vertical_expansion)
    for ticks, setter in (
        (flux_xticks, ax_flux.set_xticks),
        (flux_yticks, ax_flux.set_yticks),
        (track_xticks, ax_track.set_xticks),
        (track_yticks, ax_track.set_yticks),
    ):
        if ticks is not None:
            setter(ticks)
    for ax in (ax_flux, ax_track):
        ax.minorticks_on()
        ax.title.set_fontsize(10)

    for time in contact_times:
        ax_flux.axvline(time, color="0.65", ls=":", lw=0.8, zorder=0)
    ax_track.plot(0, 0, marker="+", color="0.4", ms=5, mew=0.7, zorder=1)
    cursor = ax_flux.axvline(relative_times[0], color="0.4", lw=0.8, alpha=0.7)
    time_label = ax_flux.text(
        0.04, 0.94, "", transform=ax_flux.transAxes, ha="left", va="top", fontsize=10
    )
    flux_trails, track_trails, flux_dots, track_dots = [], [], [], []
    markers = ("o", "s", "D")
    marker_sizes = (5.7, 5.4, 4.8)
    for color, ls, marker, ms, A, shift, full_shift in zip(
        colors,
        line_styles,
        markers,
        marker_sizes,
        magnifications,
        shifts,
        full_shifts,
    ):
        ax_flux.plot(relative_times, A, color=color, ls=ls, alpha=0.28, lw=1.15)
        ax_track.plot(
            full_shift.real,
            full_shift.imag,
            color=color,
            ls=ls,
            alpha=0.28,
            lw=1.15,
        )
        (flux_trail,) = ax_flux.plot([], [], color=color, ls=ls, lw=1.45)
        (track_trail,) = ax_track.plot([], [], color=color, ls=ls, lw=1.45)
        dot_style = dict(
            color=color, marker=marker, ms=ms, mec="white", mew=0.7, ls="none", zorder=5
        )
        (flux_dot,) = ax_flux.plot([], [], clip_on=False, **dot_style)
        (track_dot,) = ax_track.plot([], [], **dot_style)
        flux_trails.append(flux_trail)
        track_trails.append(track_trail)
        flux_dots.append(flux_dot)
        track_dots.append(track_dot)

    handles = [
        Line2D([], [], color=color, ls=ls, marker=marker, ms=4.5, label=label)
        for color, ls, marker, label in zip(colors, line_styles, markers, labels)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.525, 0.985),
        ncol=len(labels),
        handlelength=2.8,
        columnspacing=2.2,
    )

    def draw_frame(index):
        cursor.set_xdata([relative_times[index], relative_times[index]])
        time_label.set_text(rf"$t-t_0 = {relative_times[index]:+.1f}$ day")
        for A, shift, flux_trail, track_trail, flux_dot, track_dot in zip(
            magnifications, shifts, flux_trails, track_trails, flux_dots, track_dots
        ):
            flux_trail.set_data(relative_times[: index + 1], A[: index + 1])
            track_trail.set_data(shift[: index + 1].real, shift[: index + 1].imag)
            flux_dot.set_data([relative_times[index]], [A[index]])
            track_dot.set_data([shift[index].real], [shift[index].imag])
        fig.canvas.draw()
        return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    # One palette for every frame keeps the static axes and curve colors stable.
    # The last frame contains all the bright trails and supplies their colors.
    palette = draw_frame(frame_indices[-1]).quantize(colors=256)
    frames = [
        draw_frame(index).quantize(palette=palette, dither=Image.Dither.NONE)
        for index in frame_indices
    ]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / filename
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=50,
        loop=0,
        disposal=1,
        optimize=False,
    )
    plt.close(fig)
    return path


def main():
    from example.astrometry.code.astrometry import calculate_examples

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    path = save_source_animation(calculate_examples(), args.output_dir)
    print(path)


if __name__ == "__main__":
    main()
