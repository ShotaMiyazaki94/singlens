"""Figures for the annual-parallax inference comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from example._common import save_figure, style

COLORS = {"nuts": "#247BA0", "emcee": "#E07A5F"}
LABELS = {
    "t0": r"$t_0$ [d]",
    "u0": r"$u_0$",
    "t_e": r"$t_E$ [d]",
    "rho": r"$\rho$",
    "pi_e_n": r"$\pi_{E,N}$",
    "pi_e_e": r"$\pi_{E,E}$",
    "a1": r"$a_1$",
    "f_s": r"$f_s$",
    "f_b": r"$f_b$",
}


def _lens_names(report):
    names = tuple(report["sampled_parameters"])
    if not names or len(set(names)) != len(names):
        raise ValueError("sampled_parameters must contain distinct parameter names")
    return names


def _methods(report):
    rows = {row["method"]: row for row in report["results"]}
    labels = {"table": "singlens Table", "vbm": "VBMicrolensing"}
    samplers = {"nuts": "NumPyro NUTS", "emcee": "emcee"}
    return [
        dict(rows[name], label=f"{labels[rows[name]['solver']]} + {sampler}")
        for name, sampler in samplers.items()
        if name in rows
    ]


def _segments(times):
    """Do not connect the injected curve across gaps in the observing schedule."""
    spacing = np.diff(times)
    positive = spacing[spacing > 0]
    if not len(positive):
        return [np.arange(len(times))]
    gaps = np.flatnonzero(spacing > max(5 * np.median(positive), 3.0)) + 1
    return np.split(np.arange(len(times)), gaps)


def data_figure(data, report):
    """Show all observations and the finite-source peak, with injected residuals."""
    order = np.argsort(np.asarray(data.times))
    time = np.asarray(data.times)[order] - data.truth["t0"]
    observed = np.asarray(data.flux)[order]
    error = np.asarray(data.error)[order]
    injected = np.asarray(data.lens_flux)[order]
    if not (time.shape == observed.shape == error.shape == injected.shape):
        raise ValueError("Observation arrays must have identical shapes")
    if not np.all(np.isfinite(np.stack((time, observed, error, injected)))):
        raise ValueError("Observation arrays must be finite")
    if np.any(error <= 0):
        raise ValueError("Observation uncertainties must be positive")
    if np.any(observed <= 0) or np.any(injected <= 0):
        raise ValueError("Apparent magnitudes require positive observed and injected fluxes")
    baseline_magnitude = report["physical_configuration"]["baseline_magnitude"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.8, 6.2),
        sharex="col",
        gridspec_kw={"height_ratios": (3, 1)},
        layout="constrained",
    )
    spacing = np.diff(time)
    cadence = np.median(spacing[spacing > 0]) if np.any(spacing > 0) else 0.01
    peak_width = max(3 * data.truth["rho"] * data.truth["t_e"], 10 * cadence)
    selections = (np.ones(len(time), dtype=bool), np.abs(time) <= peak_width)
    for column, selected in enumerate(selections):
        top, bottom = axes[:, column]
        t, y, sigma, mean = [a[selected] for a in (time, observed, error, injected)]
        magnitude = baseline_magnitude - 2.5 * np.log10(y)
        magnitude_error = 2.5 / np.log(10) * sigma / y
        injected_magnitude = baseline_magnitude - 2.5 * np.log10(mean)
        top.errorbar(
            t,
            magnitude,
            yerr=magnitude_error,
            fmt=".",
            color="0.45",
            markersize=2.4,
            elinewidth=0.45,
            alpha=0.45,
            rasterized=True,
            label="Observations",
        )
        for index, segment in enumerate(_segments(t)):
            top.plot(
                t[segment],
                injected_magnitude[segment],
                color="0.1",
                lw=1.5,
                label="Injected mean" if index == 0 else None,
            )
        bottom.scatter(
            t,
            (y - mean) / sigma,
            s=3,
            color="0.45",
            alpha=0.5,
            rasterized=True,
        )
        bottom.axhline(0, color="0.1", lw=0.8)
        bottom.set(xlabel=r"Time from $t_0$ [d]")
        top.set_title("Observing season" if column == 0 else "Finite-source peak")
        top.ticklabel_format(axis="y", style="plain", useOffset=False)
        top.invert_yaxis()
        if column == 1:
            top.set_xlim(-peak_width, peak_width)
    axes[0, 0].set_ylabel(r"Apparent $I$ magnitude")
    axes[1, 0].set_ylabel(r"$(F-F_{\rm injected})/\sigma$")
    axes[0, 0].legend(fontsize=10)
    return fig


def sampling_figure(report):
    """Compare actual cumulative ESS and efficiency for every sampled parameter."""
    names = _lens_names(report)
    rows = _methods(report)
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.3), layout="constrained")
    for index, row in enumerate(rows):
        color = COLORS[row["method"]]
        history = row["history"]
        times = np.array([entry["elapsed_seconds"] for entry in history], dtype=float)
        ess = np.array([entry["min_ess"] for entry in history], dtype=float)
        valid = np.isfinite(times) & np.isfinite(ess) & (times >= 0) & (ess >= 0)
        axes[0].plot(times[valid], ess[valid], "o-", color=color, markersize=3, label=row["label"])
        axes[0].scatter(row["total_seconds"], row["min_ess"], s=45, color=color, zorder=3)
        offset = (index - (len(rows) - 1) / 2) * 0.2
        rates = [row["parameters"][name]["ess"] / row["total_seconds"] for name in names]
        axes[1].scatter(rates, np.arange(len(names)) + offset, color=color, s=40)
    target = report.get("target_ess", report.get("settings", {}).get("target_ess"))
    if target is not None:
        axes[0].axhline(target, color="0.55", linestyle="--", lw=0.8, zorder=0)
    axes[0].set(
        xlabel="Time including setup and warmup [s]",
        ylabel="Minimum parameter ESS",
        title="Sampling progress",
        xlim=(0, None),
        ylim=(0, None),
    )
    axes[0].legend(fontsize=11)
    axes[1].set(
        xlabel="ESS / s (including setup and warmup)",
        title="Efficiency by parameter",
        yticks=np.arange(len(names)),
        yticklabels=[LABELS.get(name, name) for name in names],
        xlim=(0, None),
    )
    axes[1].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", color="0.92", linewidth=0.7)
    return fig


def corner_figure(data, report, draws):
    """Overlay posterior shapes; thinning here is only for drawing the figure."""
    import corner

    names = _lens_names(report)
    rows = _methods(report)
    values = []
    truth = np.array([data.truth[name] for name in names], dtype=float)
    labels = [LABELS.get(name, name) for name in names]
    for row in rows:
        sample = np.column_stack(
            [np.asarray(draws[row["method"]][name]).reshape(-1) for name in names]
        )
        if not len(sample) or not np.isfinite(sample).all():
            raise ValueError("Posterior draws must be nonempty and finite")
        sample = sample[np.linspace(0, len(sample) - 1, min(5000, len(sample)), dtype=int)]
        if "t0" in names:
            sample[:, names.index("t0")] -= data.truth["t0"]
        values.append(sample)
    if "t0" in names:
        truth[names.index("t0")] = 0
        labels[names.index("t0")] = r"$t_0-t_{0,\rm injected}$ [d]"
    quantiles = np.array([np.quantile(sample, (0.001, 0.999), axis=0) for sample in values])
    low = np.minimum(quantiles[:, 0].min(axis=0), truth)
    high = np.maximum(quantiles[:, 1].max(axis=0), truth)
    margin = 0.05 * (high - low)
    if np.any(margin <= 0):
        raise ValueError("Each plotted parameter must have a nonzero posterior range")
    ranges = list(zip(low - margin, high + margin))
    fig = None
    for index, (row, sample) in enumerate(zip(rows, values)):
        fig = corner.corner(
            sample,
            fig=fig,
            labels=labels,
            range=ranges,
            bins=32,
            weights=np.full(len(sample), 1 / len(sample)),
            color=COLORS[row["method"]],
            truths=truth if index == 0 else None,
            truth_color="0.25",
            smooth=0.8,
            smooth1d=0.8,
            levels=(0.68, 0.95),
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            no_fill_contours=True,
            max_n_ticks=3,
            labelpad=0.15,
            label_kwargs={"fontsize": 12},
            hist_kwargs={"linewidth": 1.5},
            contour_kwargs={"linewidths": 1.2},
        )
    handles = [
        Line2D([], [], color=COLORS[row["method"]], lw=2, label=row["label"]) for row in rows
    ]
    handles.append(Line2D([], [], color="0.25", lw=1, label="Injected"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.93, 0.94), fontsize=16)
    return fig


def make_figures(data, report, draws, output):
    """Write PNG/PDF figures after every sampler timer has stopped."""
    style()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for name, make in (
        ("data", lambda: data_figure(data, report)),
        ("sampling", lambda: sampling_figure(report)),
        ("corner", lambda: corner_figure(data, report, draws)),
    ):
        figure = make()
        save_figure(figure, name, output)
        plt.close(figure)
