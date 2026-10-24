"""Figures for joint photometric and astrometric injection–recovery.

All centroid displays use one fixed reference: the fitted unlensed
source-plus-lens centroid at each epoch. Posterior curves vary the fitted
baseline parameters as well as the microlensing parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from chainconsumer import Chain, ChainConsumer, PlotConfig, Truth
from matplotlib.lines import Line2D
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from example._common import save_figure
from example.astrometry.code.astrometry import publication_style
from example.roman_astrometry.code.fit import RomanJointFit

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE = "#0072B2", "#D55E00"
SEASON_COLORS = ("#0072B2", "#D55E00", "#009E73", "#8A6194", "#BB9325", "#71818B")


def binned(times, values, errors, width):
    """Weighted bins for display only; the fit uses every unbinned sample."""
    times, values, errors = map(np.asarray, (times, values, errors))
    groups = np.floor((times - times.min()) / width).astype(int)
    result = []
    for group in np.unique(groups):
        selected = groups == group
        weights = errors[selected] ** -2
        weight_sum = weights.sum(axis=0)
        average = np.sum(weights * values[selected], axis=0) / weight_sum
        if values.ndim == 1:
            time_weight = weights
        else:
            time_weight = weights.sum(axis=1)
        mean_time = np.sum(time_weight * times[selected]) / time_weight.sum()
        result.append((mean_time, average, weight_sum**-0.5))
    return tuple(np.asarray(value) for value in zip(*result))


def season_mask(times, edges):
    return (times >= edges[0]) & (times <= edges[1])


def branch_sign(label):
    return r"$u_0>0$" if label == "positive" else r"$u_0<0$"


def posterior_cloud(fit, q, beta, times):
    """Use conditional baseline draws saved by the sampler, without fixing beta."""
    import jax
    import jax.numpy as jnp

    result = jax.jit(jax.vmap(lambda one_q, one_beta: fit.predict(one_q, times, beta=one_beta)))(
        jnp.asarray(q), jnp.asarray(beta)
    )
    return {key: np.asarray(value) for key, value in result.items()}


def load_observations(fit, directory, summary):
    """Plot the saved data, checking the seed, model and recorded file hashes."""
    hashes = summary.get("provenance", {}).get("synthetic_sha256", {})
    arrays = {}
    for name in ("photometry.csv", "astrometry.csv"):
        path = directory / name
        if name in hashes and hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]:
            raise ValueError(f"{name} differs from the data recorded in summary.json")
        arrays[name] = np.loadtxt(path, delimiter=",", skiprows=1)
    phot, ast = arrays["photometry.csv"], arrays["astrometry.csv"]
    np.testing.assert_allclose(phot[:, 0], fit.phot_times, rtol=0, atol=5e-8)
    np.testing.assert_allclose(ast[:, 0], fit.ast_times, rtol=0, atol=5e-8)
    np.testing.assert_allclose(
        phot[:, 1:],
        np.column_stack([fit.phot_flux, fit.phot_sigma, fit.phot_truth]),
        rtol=0,
        atol=5e-11,
    )
    np.testing.assert_allclose(
        ast[:, 1:], np.column_stack([fit.ast_xy, fit.ast_sigma, fit.ast_truth]), rtol=0, atol=5e-10
    )
    fit.phot_times, fit.phot_flux, fit.phot_sigma = phot[:, 0], phot[:, 1], phot[:, 2]
    fit.ast_times, fit.ast_xy, fit.ast_sigma = ast[:, 0], ast[:, 1:3], ast[:, 3:5]


def load_posterior(directory, summary, branch):
    """Reject stale archives or an absent/unconverged current sampling record."""
    sampling = summary["branches"][branch].get("sampling", {})
    if not sampling.get("converged", False):
        raise ValueError(f"{branch}: no converged posterior from the run in summary.json")
    path = directory / f"posterior_{branch}.npz"
    if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sampling.get(
        "posterior_sha256"
    ):
        raise ValueError(f"{branch}: posterior archive does not match the current sampling record")
    return np.load(path)


def load_bundle(directory, branch=None, draws=128):
    summary = json.loads((directory / "summary.json").read_text())
    if branch is None:
        branch = summary.get("selected_branch", "positive")
    saved = summary["branches"][branch]
    fit = RomanJointFit(**saved["settings"])
    truth_fit = RomanJointFit(seed=fit.seed, u0_sign=1)
    load_observations(fit, directory, summary)
    posterior = load_posterior(directory, summary, branch)
    q_chain, beta_chain = posterior["q"], posterior["beta"]
    q = q_chain.reshape(-1, q_chain.shape[-1])
    beta = beta_chain.reshape(-1, beta_chain.shape[-1])
    indices = np.linspace(0, len(q) - 1, min(draws, len(q)), dtype=int)
    q_draws, beta_draws = q[indices], beta[indices]
    map_q, map_beta = np.asarray(saved["map_q"]), np.asarray(saved["map_beta"])
    seasons = np.asarray(fit.season_edges)
    grid = np.concatenate([np.linspace(start, end, 161) for start, end in seasons])
    return dict(
        summary=summary,
        branch=branch,
        fit=fit,
        truth_fit=truth_fit,
        map_q=map_q,
        map_beta=map_beta,
        seasons=seasons,
        grid=grid,
        map=fit.predict(map_q, grid, beta=map_beta),
        truth=truth_fit.predict(truth_fit.q_truth, grid, beta=truth_fit.beta_truth),
        baseline=np.asarray(fit.baseline(grid, map_beta)),
        draws=posterior_cloud(fit, q_draws, beta_draws, grid),
    )


def recovery_figure(data, output):
    publication_style()
    fit, grid, seasons = data["fit"], data["grid"], data["seasons"]
    t0 = float(fit.truth["t0"])
    event = season_mask(grid, seasons[0])
    measured_event = season_mask(fit.phot_times, seasons[0])
    times = np.asarray(fit.phot_times)[measured_event]
    observed = np.asarray(fit.phot_flux)[measured_event]
    error = np.asarray(fit.phot_sigma)[measured_event]
    predicted = np.asarray(fit.predict(data["map_q"], times, beta=data["map_beta"])["flux"])
    quantiles = np.quantile(
        data["draws"]["flux"][:, event], [0.025, 0.16, 0.5, 0.84, 0.975], axis=0
    )
    fig = plt.figure(figsize=(11.8, 5.0))
    gs = fig.add_gridspec(
        2, 2, width_ratios=(1.15, 1), height_ratios=(3.5, 1), hspace=0.08, wspace=0.28
    )
    ax = fig.add_subplot(gs[0, 0])
    residual_ax = fig.add_subplot(gs[1, 0], sharex=ax)
    geometry = fig.add_subplot(gs[:, 1])
    bt, bf, be = binned(times, observed, error, 0.5)
    ax.plot(times - t0, observed, ".", ms=0.6, color=".75", alpha=0.3, rasterized=True)
    ax.errorbar(
        bt - t0,
        bf,
        be,
        fmt=".",
        color=".3",
        ms=3,
        elinewidth=0.6,
        label="Synthetic data, 0.5-day bins",
    )
    ax.fill_between(
        grid[event] - t0,
        quantiles[0],
        quantiles[4],
        color=BLUE,
        alpha=0.15,
        lw=0,
        label="95% model interval",
    )
    ax.fill_between(grid[event] - t0, quantiles[1], quantiles[3], color=BLUE, alpha=0.3, lw=0)
    ax.plot(grid[event] - t0, quantiles[2], color=BLUE, label="Posterior median")
    ax.plot(
        grid[event] - t0,
        np.asarray(data["truth"]["flux"])[event],
        "--",
        color=".15",
        lw=1.1,
        label="Injected model",
    )
    ax.set(ylabel="Flux / injected baseline flux", xlim=seasons[0] - t0)
    ax.set_title("Event-season photometry", loc="left")
    ax.tick_params(labelbottom=False)
    ax.legend(fontsize=8, loc="upper right")
    rt, rr, re = binned(times, observed - predicted, error, 0.5)
    residual_ax.errorbar(rt - t0, rr * 1000, re * 1000, fmt=".", ms=3, color=".35", elinewidth=0.6)
    residual_ax.axhline(0, color=BLUE, lw=0.8)
    residual_ax.set(xlabel=r"Days from injected $t_0$", ylabel=r"$\Delta F$ ($10^{-3}$)")

    baseline = data["baseline"]
    model = (np.asarray(data["map"]["centroid_mas"]) - baseline) * 1000
    truth = (np.asarray(data["truth"]["centroid_mas"]) - baseline) * 1000
    clouds = (data["draws"]["centroid_mas"] - baseline[None, :, :]) * 1000
    for cloud in clouds[::2]:
        for edges in seasons:
            keep = season_mask(grid, edges)
            geometry.plot(cloud[keep, 0], cloud[keep, 1], color=BLUE, alpha=0.045, lw=0.65)
    for index, edges in enumerate(seasons):
        keep = season_mask(grid, edges)
        geometry.plot(truth[keep, 0], truth[keep, 1], "--", color=".1", lw=1.1)
        geometry.plot(model[keep, 0], model[keep, 1], color=BLUE, lw=1.4)
        obs_keep = season_mask(fit.ast_times, edges)
        obs_time = np.asarray(fit.ast_times)[obs_keep]
        fixed_baseline = np.asarray(fit.baseline(obs_time, data["map_beta"]))
        transformed = np.asarray(fit.ast_xy)[obs_keep] - fixed_baseline
        bin_width = 6 if index == 0 else edges[1] - edges[0] + 1
        _, xy, xy_error = binned(
            obs_time, transformed, np.asarray(fit.ast_sigma)[obs_keep], bin_width
        )
        geometry.errorbar(
            xy[:, 0] * 1000,
            xy[:, 1] * 1000,
            xerr=xy_error[:, 0] * 1000,
            yerr=xy_error[:, 1] * 1000,
            fmt="o",
            ms=2.6,
            elinewidth=0.55,
            color=SEASON_COLORS[index],
            alpha=0.75,
        )
        if index == 0:
            k = np.flatnonzero(keep)
            mid = len(k) // 3
            geometry.annotate(
                "",
                xy=model[k[mid + 3]],
                xytext=model[k[mid]],
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.3),
            )
    geometry.axhline(0, color=".83", lw=0.6, zorder=-1)
    geometry.axvline(0, color=".83", lw=0.6, zorder=-1)
    geometry.set_ylim(top=330)
    angular_span = np.ptp(geometry.get_ylim())
    east_center = np.mean(geometry.get_xlim())
    geometry.set_xlim(east_center - angular_span / 2, east_center + angular_span / 2)
    geometry.set(
        xlabel=r"East offset ($\mu$as)",
        ylabel=r"North offset ($\mu$as)",
        aspect="equal",
    )
    geometry.set_title("Centroid geometry · all six seasons", loc="left")
    handles = [
        Line2D([], [], color=BLUE, lw=1.4, label="MAP centroid"),
        Line2D([], [], color=".1", ls="--", label="Injected centroid"),
        Line2D([], [], color=BLUE, alpha=0.3, lw=5, label="Posterior draws; baseline varies"),
    ]
    geometry.legend(handles=handles, loc="best", fontsize=7.5)
    fig.suptitle(f"Joint Roman-like injection–recovery · {branch_sign(data['branch'])}", y=0.99)
    fig.text(
        0.53,
        0.01,
        "Origin at each epoch: fitted unlensed source + lens centroid.\n"
        "Season 1: 6-day bins; later seasons: means. Bars: measurement errors.",
        fontsize=8,
    )
    fig.subplots_adjust(bottom=0.16, top=0.89, left=0.075, right=0.985)
    save_figure(fig, "recovery", output)
    plt.close(fig)


def astrometry_figure(data, output):
    publication_style()
    fit, seasons, grid = data["fit"], data["seasons"], data["grid"]
    t0 = float(fit.truth["t0"])
    ast_times = np.asarray(fit.ast_times)
    map_at_obs = fit.predict(data["map_q"], ast_times, beta=data["map_beta"])
    baseline_at_obs = np.asarray(fit.baseline(ast_times, data["map_beta"]))
    corrected = (np.asarray(fit.ast_xy) - baseline_at_obs) * 1000
    model = (np.asarray(data["map"]["centroid_mas"]) - data["baseline"]) * 1000
    truth = (np.asarray(data["truth"]["centroid_mas"]) - data["baseline"]) * 1000
    cloud = (data["draws"]["centroid_mas"] - data["baseline"][None, :, :]) * 1000
    quantiles = np.quantile(cloud, [0.025, 0.16, 0.84, 0.975], axis=0)
    standardized = (np.asarray(fit.ast_xy) - np.asarray(map_at_obs["centroid_mas"])) / fit.ast_sigma
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11.8, 5.5),
        sharex=True,
        gridspec_kw={"height_ratios": (1, 1, 0.7), "hspace": 0.12},
    )
    colors = (BLUE, ORANGE)
    for coordinate, ax in enumerate(axes[:2]):
        for index, edges in enumerate(seasons):
            keep = season_mask(grid, edges)
            obs_keep = season_mask(ast_times, edges)
            t, y, error = binned(
                ast_times[obs_keep],
                corrected[obs_keep, coordinate],
                np.asarray(fit.ast_sigma)[obs_keep, coordinate] * 1000,
                3,
            )
            ax.errorbar(
                (t - t0) / 365.25, y, error, fmt=".", color=".45", elinewidth=0.6, ms=3, alpha=0.8
            )
            x = (grid[keep] - t0) / 365.25
            ax.fill_between(
                x,
                quantiles[0, keep, coordinate],
                quantiles[3, keep, coordinate],
                color=colors[coordinate],
                alpha=0.15,
                lw=0,
            )
            ax.fill_between(
                x,
                quantiles[1, keep, coordinate],
                quantiles[2, keep, coordinate],
                color=colors[coordinate],
                alpha=0.3,
                lw=0,
            )
            ax.plot(x, model[keep, coordinate], color=colors[coordinate], lw=1.3)
            ax.plot(x, truth[keep, coordinate], color=".1", ls="--", lw=0.9)
            if coordinate == 0:
                ax.text(
                    np.mean(edges - t0) / 365.25,
                    0.98,
                    str(index + 1),
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                )
        direction = ("E", "N")[coordinate]
        ax.set_ylabel(
            rf"$\theta_{direction}-b_{{{direction},\mathrm{{MAP}}}}$" + "\n" + r"($\mu$as)"
        )
        ax.axhline(0, lw=0.5, color=".7", zorder=-1)
    for coordinate, name in enumerate(("East", "North")):
        axes[2].plot(
            (ast_times - t0) / 365.25,
            standardized[:, coordinate],
            ".",
            ms=2.2,
            alpha=0.55,
            color=colors[coordinate],
            label=name,
        )
    axes[2].axhline(0, color=".4", lw=0.7)
    for y in (-3, 3):
        axes[2].axhline(y, color=".7", ls=":", lw=0.7)
    axes[2].set(
        xlabel=r"Years from injected $t_0$",
        ylabel=r"Residual / $\sigma$",
        xlim=((seasons[0, 0] - t0) / 365.25 - 0.05, (seasons[-1, 1] - t0) / 365.25 + 0.05),
    )
    axes[2].legend(ncol=2, fontsize=8, loc="upper right")
    handles = [
        Line2D([], [], color=".4", marker=".", ls="none", label="Synthetic data, 3-day bins"),
        Line2D([], [], color=BLUE, label="MAP centroid"),
        Line2D([], [], color=".1", ls="--", label="Injected centroid"),
        Line2D([], [], color=BLUE, alpha=0.25, lw=6, label="68% / 95% model intervals"),
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 0.955), fontsize=8
    )
    fig.suptitle(
        "Six-season astrometric fit · one fitted baseline subtracted from data and every model",
        y=0.995,
        fontsize=11,
    )
    fig.text(
        0.08,
        0.01,
        "Bands include uncertainty in baseline position, proper motion and parallax. Bottom: individual, unbinned residuals.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.865, bottom=0.13)
    save_figure(fig, "astrometry_fit", output)
    plt.close(fig)


CORNER_PARAMETERS = (
    "theta_e_mas",
    "pi_e_n",
    "pi_e_e",
    "u0",
    "t_e_day",
    "lens_to_source_flux",
    "mass_msun",
)
CORNER_LABELS = dict(
    theta_e_mas=r"$\theta_E$ (mas)",
    pi_e_n=r"$\pi_{E,N}$",
    pi_e_e=r"$\pi_{E,E}$",
    u0=r"$u_0$",
    t_e_day=r"$t_E$ (day)",
    lens_to_source_flux=r"$g=F_L/F_S$",
    mass_msun=r"$M_L/M_\odot$",
)


def physical_samples(fit, posterior):
    samples = {}
    for name in CORNER_PARAMETERS:
        if name in posterior.files:
            samples[name] = posterior[name].reshape(-1)
    if len(samples) < len(CORNER_PARAMETERS):
        q = posterior["q"].reshape(-1, posterior["q"].shape[-1])
        derived = fit.physical(q)
        samples = {name: np.asarray(derived[name]) for name in CORNER_PARAMETERS}
    return samples


def corner_figure(data, directory, output, *, overlay=False):
    publication_style()
    branches = [data["branch"]]
    if overlay:
        branches = ["positive", "negative"]
    consumer = ChainConsumer()
    labels = dict(CORNER_LABELS)
    if len(branches) > 1:
        labels["u0"] = r"$|u_0|$"
    extents = {name: [] for name in CORNER_PARAMETERS}
    for branch in branches:
        fit = RomanJointFit(**data["summary"]["branches"][branch]["settings"])
        posterior = load_posterior(directory, data["summary"], branch)
        samples = physical_samples(fit, posterior)
        if len(branches) > 1:
            samples["u0"] = abs(samples["u0"])
        for name, values in samples.items():
            extents[name].extend(np.quantile(values, [0.0005, 0.9995]))
        consumer.add_chain(
            Chain(
                samples=pd.DataFrame(samples),
                name=branch_sign(branch),
                color=BLUE if branch == "positive" else ORANGE,
                kde=1.0,
                bins=45,
                sigmas=[0, float(ndtri(0.84)), float(ndtri(0.975))],
                shade=len(branches) == 1,
                shade_alpha=0.18,
                bar_shade=False,
                linewidth=1.1,
                plot_cloud=False,
            )
        )
    truth = {
        name: float(data["truth_fit"].physical(data["truth_fit"].q_truth)[name])
        for name in CORNER_PARAMETERS
    }
    if len(branches) > 1:
        truth["u0"] = abs(truth["u0"])
    bounds = {}
    for name, values in extents.items():
        low, high = min(*values, truth[name]), max(*values, truth[name])
        margin = 0.07 * (high - low)
        bounds[name] = (low - margin, high + margin)
    consumer.add_truth(
        Truth(location=truth, color=".15", line_style="--", line_width=0.9, name="Injected value")
    )
    consumer.set_plot_config(
        PlotConfig(
            labels=labels,
            extents=bounds,
            summarise=False,
            sigma2d=False,
            max_ticks=3,
            serif=True,
            label_font_size=11,
            tick_font_size=8,
            spacing=0.08,
            show_legend=False,
        )
    )
    fig = consumer.plotter.plot(columns=list(CORNER_PARAMETERS), figsize=(11.7, 11.7))
    fig.text(0.66, 0.93, "Joint photometry + astrometry", fontsize=15)
    fig.text(
        0.66,
        0.89,
        "68% / 95% posterior contours\nDashed black: injected values",
        fontsize=11,
        linespacing=1.6,
    )
    for index, branch in enumerate(branches):
        samples = physical_samples(
            RomanJointFit(**data["summary"]["branches"][branch]["settings"]),
            load_posterior(directory, data["summary"], branch),
        )
        low, med, high = np.quantile(samples["mass_msun"], [0.16, 0.5, 0.84])
        fig.text(
            0.66,
            0.82 - 0.065 * index,
            branch_sign(branch)
            + rf": $M_L={med:.2f}_{{-{med - low:.2f}}}^{{+{high - med:.2f}}}\,M_\odot$",
            fontsize=13,
            color=BLUE if branch == "positive" else ORANGE,
        )
    if len(branches) > 1:
        difference = data["summary"]["branches"]["negative"]["delta_map_log_posterior"]
        fig.text(
            0.66,
            0.65,
            r"$|u_0|$ is shown to compare the signs."
            + "\nEach branch normalized separately.\nNo posterior branch weights assigned.\n"
            + rf"Negative branch: $\Delta\log p_{{\rm MAP}}={difference:.1f}$ below best.",
            fontsize=10,
            linespacing=1.6,
        )
    save_figure(fig, "branch_comparison" if overlay else "corner", output)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "outputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--branch", choices=("positive", "negative"))
    parser.add_argument("--draws", type=int, default=128)
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Compare available sign branches without assigning branch weights",
    )
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    data = load_bundle(args.input, args.branch, args.draws)
    recovery_figure(data, output)
    astrometry_figure(data, output)
    corner_figure(data, args.input, output)
    if args.overlay:
        corner_figure(data, args.input, output, overlay=True)


if __name__ == "__main__":
    main()
