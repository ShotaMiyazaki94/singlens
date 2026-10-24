"""Publication figures for the active-star / free-floating-planet example.

Read the saved observations and conditional posterior predictions. The
season-wide variability panel subtracts the injected lens-only magnitude;
the thin residual strips subtract the full fitted lens-plus-GP prediction.
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
from scipy.special import ndtri

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from example._common import save_figure
from example.astrometry.code.astrometry import publication_style

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "active_ffp"
BLUE, ORANGE = "#0072B2", "#D55E00"


def read_checked(path, expected):
    """Require each product to match the completed run's recorded hash."""
    if not path.exists():
        raise FileNotFoundError(f"Missing saved experiment product: {path}")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"Missing or invalid recorded SHA-256: {path.name}")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Saved experiment hash mismatch: {path.name}")
    return path


def magnitude(flux, zeropoint):
    values = np.asarray(flux)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Magnitude figures require finite, positive flux")
    return zeropoint - 2.5 * np.log10(values)


def bin_measurements(times, values, errors, width):
    """Display-only inverse-variance binning; quoted errors exclude GP covariance."""
    groups = np.floor(np.round((times - times.min()) / width, 9)).astype(int)
    weights = errors**-2
    sums = np.bincount(groups, weights=weights)
    occupied = sums > 0
    coordinates = [
        np.bincount(groups, weights=weights * value)[occupied] / sums[occupied]
        for value in (times, values)
    ]
    return *coordinates, sums[occupied] ** -0.5


def magnitude_predictions(data):
    relative = data["times"] - data["truth"]["t0"]
    observed = magnitude(data["flux"], data["zeropoint"])
    lens = magnitude(data["lens_flux"], data["zeropoint"])
    total = magnitude(data["lens_flux"] + data["gp_signal"], data["zeropoint"])
    paths = magnitude(data["draw_total_flux"], data["zeropoint"])
    quantiles = np.quantile(paths, [0.025, 0.16, 0.5, 0.84, 0.975], axis=0)
    fitted_flux = 10 ** ((data["zeropoint"] - quantiles[2]) / 2.5)
    sigma = 2.5 / np.log(10) * data["error"] / fitted_flux
    return relative, observed, lens, total, quantiles, sigma


def rotational_covariance(lag, params):
    """Analytic covariance of the two damped SHO harmonics, in source-flux units."""
    lag = np.abs(np.asarray(lag))
    decay = 1 / params["gp_scale"]
    variance = (params["baseline"] * params["source_fraction"] * params["gp_rms"]) ** 2
    covariance = np.zeros_like(lag)
    for harmonic, fraction in ((1, 0.8), (2, 0.2)):
        frequency = 2 * np.pi * harmonic / params["gp_period"]
        phase = frequency * lag
        covariance += (
            variance
            * fraction
            * np.exp(-decay * lag)
            * (np.cos(phase) + decay / frequency * np.sin(phase))
        )
    return covariance


def interpolation_rms(params, spacing):
    """Prior RMS of g(t) minus its linearly interpolated endpoints.

    The full three-location covariance expression is evaluated at 65 positions
    across the largest cadence gap. Extended precision limits cancellation.
    This bounds conditional covariance, not every conditional draw or mean.
    """
    fraction = np.linspace(0, 1, 65, dtype=np.longdouble)
    p = {name: np.longdouble(value) for name, value in params.items()}
    gap = np.longdouble(spacing)
    k0 = rotational_covariance(np.longdouble(0), p)
    variance = (
        k0 * (1 + (1 - fraction) ** 2 + fraction**2)
        + 2 * fraction * (1 - fraction) * rotational_covariance(gap, p)
        - 2 * (1 - fraction) * rotational_covariance(fraction * gap, p)
        - 2 * fraction * rotational_covariance((1 - fraction) * gap, p)
    )
    return float(np.sqrt(np.maximum(variance, 0).max()))


def dense_event_predictions(data):
    """Recompute the lens on a dense grid; interpolate only the intrinsic GP.

    Parameter draws use the same deterministic indices as save_predictive.
    Their exposure-averaged lens curves must reproduce the saved predictions
    before they can be paired with the saved conditional GP paths.
    """
    import jax
    import jax.numpy as jnp

    import singlens as sl
    from example.inference.code.active_ffp import NAMES, build_gp

    jax.config.update("jax_enable_x64", True)
    report = data["report"]
    times = data["times"]
    half_width = 3 * data["truth"]["t_e"]
    event = abs(times - data["truth"]["t0"]) <= half_width
    grid = np.unique(
        np.r_[
            data["truth"]["t0"] + np.linspace(-half_width, half_width, 501),
            times[event],
        ]
    )
    nodes, weights = np.polynomial.legendre.leggauss(3)
    nodes = jnp.asarray(nodes * report["configuration"]["exposure_seconds"] / (2 * 86400))
    weights = jnp.asarray(weights / 2)
    finite_model = sl.Model(source="linear", solver="fft")
    point_model = sl.Model(source="point")

    def magnification(params, at, *, point=False):
        coordinates = dict(t0=params["t0"], u0=params["u0"], t_e=params["t_e"])
        if not point:
            coordinates.update(rho=params["rho"], a1=report["fixed"]["a1"])
        exposure_times = jnp.asarray(at)[:, None] + nodes
        model = point_model if point else finite_model
        amp = model.magnification(exposure_times.ravel(), sl.Parameters(**coordinates))
        return amp.reshape(exposure_times.shape) @ weights

    finite_amp = jax.jit(magnification)
    point_amp = jax.jit(lambda p, t: magnification(p, t, point=True))
    posterior = data["posterior"]
    chain_shape = posterior["q"].shape[:-1]
    count = data["draw_total_flux"].shape[0]
    indices = np.linspace(0, np.prod(chain_shape) - 1, count, dtype=int)
    parameters = {name: posterior[name].reshape(-1)[indices] for name in NAMES}

    def evaluate_paired(at):
        return np.asarray(
            jax.jit(lambda p: jax.lax.map(lambda v: magnification(v, at), p))(
                {name: jnp.asarray(value) for name, value in parameters.items()}
            )
        )

    observed_amp = evaluate_paired(times)
    dense_amp = evaluate_paired(grid)
    source = (parameters["baseline"] * parameters["source_fraction"])[:, None]
    blend = parameters["baseline"][:, None] - source
    recomputed = source * observed_amp + blend
    np.testing.assert_allclose(recomputed, data["draw_lens_flux"], rtol=2e-12, atol=2e-12)
    intrinsic = (data["draw_total_flux"] - data["draw_lens_flux"]) / observed_amp
    interpolated = np.asarray([np.interp(grid, times, path) for path in intrinsic])
    total = source * dense_amp + blend + dense_amp * interpolated
    truth = data["truth"]
    truth_lens = truth["baseline"] * truth["source_fraction"] * np.asarray(
        finite_amp(truth, grid)
    ) + truth["baseline"] * (1 - truth["source_fraction"])

    point = report["physical_checks"]["finite_source"]["comparison"]["actual_data_gp_pspl"][
        "point_map_parameters"
    ]
    point_observed_amp = np.asarray(point_amp(point, times))
    point_dense_amp = np.asarray(point_amp(point, grid))
    point_source = point["baseline"] * point["source_fraction"]
    point_blend = point["baseline"] - point_source
    point_lens = point_source * point_observed_amp + point_blend
    np.testing.assert_allclose(
        point_lens, data["point_source"]["point_lens_flux"], rtol=2e-12, atol=2e-12
    )
    point_intrinsic = (data["point_source"]["point_mean_flux"] - point_lens) / point_observed_amp
    point_total = point_source * point_dense_amp + point_blend
    point_total += point_dense_amp * np.interp(grid, times, point_intrinsic)

    # Conservatively use the largest amplification and smallest quoted flux
    # error, rather than interpolating error bars or modifying observations.
    gap = float(np.diff(times).max())
    white_sigma = float(data["error"].min())
    rms_ratios = [
        interpolation_rms({name: value[i] for name, value in parameters.items()}, gap)
        * max(dense_amp[i].max(), observed_amp[i].max())
        / white_sigma
        for i in range(count)
    ]
    point_ratio = (
        interpolation_rms(point, gap)
        * max(point_dense_amp.max(), point_observed_amp.max())
        / white_sigma
    )

    def check_conditional_mean(params, amp, dense, saved_mean):
        """Independent dense GP conditioning; no interpolation in this check."""
        fs = params["baseline"] * params["source_fraction"]
        lens = fs * amp + params["baseline"] - fs

        @jax.jit
        def solve():
            gp = build_gp(jnp.asarray(times), jnp.asarray(data["error"] / amp), params)
            residual = jnp.asarray((data["flux"] - lens) / amp)
            solved = gp.solver.solve_triangular(residual)
            solved = gp.solver.solve_triangular(solved, transpose=True)
            return solved, gp.kernel.matmul(jnp.asarray(times), solved)

        solved, observed_mean = (np.asarray(value) for value in solve())
        if saved_mean is not None:
            np.testing.assert_allclose(
                lens + amp * observed_mean, saved_mean, rtol=2e-10, atol=2e-10
            )
        exact = np.concatenate(
            [
                rotational_covariance(block[:, None] - times[None, :], params) @ solved
                for block in np.array_split(grid, 10)
            ]
        )
        error = dense * (exact - np.interp(grid, times, observed_mean))
        return dict(
            max_absolute_flux_error=float(np.max(np.abs(error))),
            max_absolute_error_over_min_white_sigma=float(np.max(np.abs(error)) / white_sigma),
        )

    map_params = report["map_physical"]
    conditional_checks = dict(
        finite_source_map=check_conditional_mean(
            map_params,
            np.asarray(finite_amp(map_params, times)),
            np.asarray(finite_amp(map_params, grid)),
            data["map_total_flux"],
        ),
        point_source_map=check_conditional_mean(
            point, point_observed_amp, point_dense_amp, data["point_source"]["point_mean_flux"]
        ),
    )
    threshold = 0.01
    maximum = float(max(max(rms_ratios), point_ratio))
    if maximum >= threshold or any(
        result["max_absolute_error_over_min_white_sigma"] >= threshold
        for result in conditional_checks.values()
    ):
        raise ValueError("Dense GP interpolation error exceeds 1% of the quoted white-noise sigma")
    validation = dict(
        method="Exact three-node exposure-averaged lens curves; linearly interpolated intrinsic GP only",
        dense_grid_points=len(grid),
        posterior_paths=count,
        posterior_flat_indices=indices.tolist(),
        max_observation_lens_flux_pairing_error=float(
            np.max(abs(recomputed - data["draw_lens_flux"]))
        ),
        largest_cadence_gap_days=gap,
        interpolation_fractions=65,
        minimum_white_noise_sigma_flux=white_sigma,
        max_prior_interpolation_rms_over_min_white_sigma=maximum,
        posterior_max_prior_interpolation_rms_over_min_white_sigma=float(max(rms_ratios)),
        point_map_prior_interpolation_rms_over_min_white_sigma=float(point_ratio),
        threshold_over_white_sigma=threshold,
        exact_conditional_mean_checks=conditional_checks,
        interpretation=(
            "The prior covariance bounds conditional covariance of the interpolation error. "
            "Its RMS is not a sample-by-sample or conditional-mean bound. Dense conditional "
            "GP means are therefore checked independently at both MAP solutions. "
            "No observation, error bar, likelihood, or posterior sample is changed."
        ),
        provenance=dict(
            plot_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            posterior_sha256=report["sampling"]["posterior_sha256"],
            predictive_sha256=report["provenance"]["predictive_sha256"],
            point_source_comparison_sha256=report["provenance"]["point_source_comparison_sha256"],
        ),
    )
    return dict(
        times=grid,
        draw_total_flux=total,
        truth_lens_flux=truth_lens,
        point_total_flux=point_total,
        validation=validation,
    )


def model_band(ax, time, quantiles, *, reference=0, scale=1, label=None):
    lo95, lo68, median, hi68, hi95 = (quantiles - reference) * scale
    ax.fill_between(time, lo95, hi95, color=BLUE, alpha=0.15, lw=0, label=label)
    ax.fill_between(time, lo68, hi68, color=BLUE, alpha=0.3, lw=0)
    ax.plot(time, median, color=BLUE, lw=1.1, label="Finite source + GP median")


def recovery_figure(data, output):
    """Three shared-time panels: actual light curve, stellar variation, residuals."""
    publication_style()
    time, observed, lens, total, quantiles, sigma = magnitude_predictions(data)
    residual = 1000 * (observed - quantiles[2])
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12.3, 6.1),
        sharex=True,
        gridspec_kw={"height_ratios": (1.55, 1.4, 0.65), "hspace": 0.10},
    )
    light, variability, residual_ax = axes
    light.plot(
        time,
        observed,
        ".",
        color=".65",
        ms=0.9,
        alpha=0.25,
        rasterized=True,
        label="Individual observations",
    )
    model_band(light, time, quantiles)
    light.plot(time, lens, "--", color=ORANGE, lw=1, label="Injected lens only")
    light.set_ylabel("F146 (AB mag)")
    light.invert_yaxis()
    light.legend(loc="upper right", ncol=3, fontsize=8)

    variation = 1000 * (observed - lens)
    variability.plot(time, variation, ".", color=".65", ms=0.9, alpha=0.18, rasterized=True)
    bt, by, be = bin_measurements(time, variation, 1000 * sigma, 0.1)
    variability.errorbar(
        bt,
        by,
        be,
        fmt=".",
        color=".35",
        ms=2,
        elinewidth=0.4,
        alpha=0.65,
        label="Data, 0.1-day bins",
    )
    variability.plot(
        time, 1000 * (total - lens), "--", color=".15", lw=0.9, label="Injected variability"
    )
    model_band(
        variability, time, quantiles, reference=lens, scale=1000, label="68% / 95% model intervals"
    )
    variability.set_ylabel("Magnitude − injected lens\n(mmag)")
    variability.legend(loc="upper right", ncol=4, fontsize=7.5)

    residual_ax.plot(time, residual, ".", color=".6", ms=0.8, alpha=0.2, rasterized=True)
    bt, by, be = bin_measurements(time, residual, 1000 * sigma, 0.1)
    residual_ax.errorbar(bt, by, be, fmt=".", color=".3", ms=1.8, elinewidth=0.4)
    residual_ax.axhline(0, color=BLUE, lw=0.7)
    residual_ax.set(
        xlabel=r"Days from injected $t_0$",
        ylabel="Data − model\n(mmag)",
        xlim=(time.min(), time.max()),
    )
    fig.suptitle(
        "Active K dwarf + free-floating planet · joint event and stellar-GP fit",
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.085,
        0.01,
        "Bands include parameter and conditional-GP uncertainty, without new measurement noise. Bin errors include measurement noise only.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.12, top=0.94)
    save_figure(fig, "recovery", output)
    plt.close(fig)


def event_zoom_figure(data, output):
    """Actual exposures beside dense finite-source and refitted point predictions."""
    publication_style()
    time, observed, _, _, quantiles, sigma = magnitude_predictions(data)
    half_width = 3 * data["truth"]["t_e"]
    event = abs(time) <= half_width
    dense = dense_event_predictions(data)
    (output / "plot_validation.json").write_text(json.dumps(dense["validation"], indent=2) + "\n")
    grid = 24 * (dense["times"] - data["truth"]["t0"])
    dense_quantiles = np.quantile(
        magnitude(dense["draw_total_flux"], data["zeropoint"]),
        [0.025, 0.16, 0.5, 0.84, 0.975],
        axis=0,
    )
    point = magnitude(dense["point_total_flux"], data["zeropoint"])
    lens = magnitude(dense["truth_lens_flux"], data["zeropoint"])
    time, observed, quantiles, sigma = (
        24 * time[event],
        observed[event],
        quantiles[:, event],
        sigma[event],
    )
    fig, (light, residual_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 4.7),
        sharex=True,
        gridspec_kw={"height_ratios": (3.5, 1), "hspace": 0.08},
    )
    light.errorbar(
        time,
        observed,
        sigma,
        fmt=".",
        color=".3",
        ms=2.6,
        alpha=0.7,
        elinewidth=0.5,
        zorder=4,
        label="Individual observations",
    )
    model_band(light, grid, dense_quantiles, label="68% / 95% model intervals")
    light.plot(grid, lens, ":", color=".25", lw=1, label="Injected finite-source lens only")
    light.plot(grid, point, "--", color=ORANGE, lw=1.2, label="Refitted point source + GP (MAP)")
    light.set_ylabel("F146 (AB mag)")
    light.invert_yaxis()
    light.legend(loc="upper left", fontsize=8)
    residual_ax.errorbar(
        time,
        1000 * (observed - quantiles[2]),
        1000 * sigma,
        fmt=".",
        color=".4",
        ms=1.8,
        elinewidth=0.4,
        alpha=0.65,
        label="Data − FS+GP median",
    )
    residual_ax.axhline(0, color=BLUE, lw=0.7)
    residual_ax.plot(
        grid,
        1000 * (point - dense_quantiles[2]),
        "--",
        color=ORANGE,
        lw=1.2,
        label="Point+GP − FS+GP median",
    )
    residual_ax.legend(loc="upper right", fontsize=7.5)
    residual_ax.set(
        xlabel=r"Hours from injected $t_0$",
        ylabel="Relative to FS+GP\n(mmag)",
        xlim=(-24 * half_width, 24 * half_width),
    )
    fig.suptitle(
        "Finite-source event · individual exposures and refitted point-source model",
        fontsize=12,
        y=0.99,
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.14, top=0.92)
    save_figure(fig, "event_zoom", output)
    plt.close(fig)


def corner_figure(data, output):
    """Show event, blending and variability correlations with injected truth marks."""
    publication_style()
    selected = data["corner_parameters"]
    samples = {name: np.asarray(data["posterior"][name]).reshape(-1) for name in selected}
    injected = {name: float(data["truth"][name]) for name in selected}
    bounds = {}
    for name, values in samples.items():
        low, high = np.quantile(values, [0.0005, 0.9995])
        low, high = min(low, injected[name]), max(high, injected[name])
        margin = 0.07 * (high - low)
        bounds[name] = (low - margin, high + margin)
    consumer = ChainConsumer()
    consumer.add_chain(
        Chain(
            samples=pd.DataFrame(samples),
            name="Joint posterior",
            color=BLUE,
            kde=1.0,
            bins=45,
            sigmas=[0, float(ndtri(0.84)), float(ndtri(0.975))],
            shade=True,
            shade_alpha=0.15,
            bar_shade=False,
            linewidth=1,
            plot_cloud=False,
        )
    )
    consumer.add_truth(Truth(location=injected, color=".2", line_style="--", line_width=0.8))
    consumer.set_plot_config(
        PlotConfig(
            labels=data["corner_labels"],
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
    fig = consumer.plotter.plot(columns=selected, figsize=(13.2, 12.6))
    fig.text(0.64, 0.94, "Microlensing + active-star variability", fontsize=13)
    fig.text(
        0.64,
        0.885,
        "68% / 95% posterior contours\nDashed black: injected values",
        fontsize=11,
        linespacing=1.6,
    )
    save_figure(fig, "corner", output)
    plt.close(fig)


def load_results(directory=OUTPUT):
    """Read observations and posterior products from the same converged run."""
    directory = Path(directory)
    report = json.loads((directory / "summary.json").read_text())
    sampling = report.get("sampling", {})
    if not sampling.get("converged", False):
        raise ValueError("No converged posterior belongs to the current summary.json")
    provenance = report.get("provenance", {})
    with np.load(
        read_checked(directory / "posterior.npz", sampling.get("posterior_sha256"))
    ) as saved:
        posterior = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(
        read_checked(directory / "predictive.npz", provenance.get("predictive_sha256"))
    ) as saved:
        predictions = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(
        read_checked(
            directory / "point_source_comparison.npz",
            provenance.get("point_source_comparison_sha256"),
        )
    ) as saved:
        point_source = {name: np.asarray(saved[name]) for name in saved.files}
    observations = np.genfromtxt(
        read_checked(
            directory / "photometry.csv",
            provenance.get("synthetic_sha256", {}).get("photometry.csv"),
        ),
        delimiter=",",
        names=True,
    )
    for key, column in (
        ("times", "time_day"),
        ("observed_flux", "flux"),
        ("sigma_flux", "sigma_flux"),
        ("truth_lens_flux", "truth_lens_flux"),
        ("truth_gp_flux", "truth_gp_flux"),
    ):
        np.testing.assert_allclose(predictions[key], observations[column], rtol=5e-13, atol=5e-14)
    np.testing.assert_allclose(
        point_source["times"], observations["time_day"], rtol=5e-13, atol=5e-14
    )
    for key in ("point_lens_flux", "point_mean_flux", "noiseless_best_point_flux"):
        if point_source[key].shape != observations["time_day"].shape:
            raise ValueError(f"Point-source prediction shape mismatch: {key}")
    truth = dict(report["truth"])
    truth["gp_rms_percent"] = 100 * truth["gp_rms"]
    posterior["gp_rms_percent"] = 100 * posterior["gp_rms"]
    truth["t0_offset_minutes"] = 0.0
    posterior["t0_offset_minutes"] = 1440 * (posterior["t0"] - truth["t0"])
    truth["t_e_hours"] = 24 * truth["t_e"]
    posterior["t_e_hours"] = 24 * posterior["t_e"]
    truth["baseline_offset_ppt"] = 0.0
    posterior["baseline_offset_ppt"] = 1000 * (posterior["baseline"] / truth["baseline"] - 1)
    labels = dict(
        t0_offset_minutes=r"$t_0-t_{0,\mathrm{inj}}$ (min)",
        u0=r"$u_0$",
        t_e_hours=r"$t_E$ (hour)",
        rho=r"$\rho$",
        baseline_offset_ppt=r"$10^3(F_0/F_{0,\mathrm{inj}}-1)$",
        source_fraction=r"$f_s/(f_s+f_b)$",
        gp_rms_percent=r"GP RMS (%)",
        gp_scale=r"$\tau_{\rm GP}$ (day)",
        gp_period=r"$P_{\rm rot}$ (day)",
    )
    return dict(
        report=report,
        posterior=posterior,
        truth=truth,
        zeropoint=report["configuration"]["baseline_mag_ab"],
        times=observations["time_day"],
        flux=observations["flux"],
        error=observations["sigma_flux"],
        lens_flux=observations["truth_lens_flux"],
        gp_signal=observations["truth_gp_flux"],
        draw_total_flux=predictions["draw_total_flux"],
        draw_lens_flux=predictions["draw_lens_flux"],
        map_total_flux=predictions["map_total_flux"],
        point_source=point_source,
        corner_parameters=list(labels),
        corner_labels=labels,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=OUTPUT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.input
    output.mkdir(parents=True, exist_ok=True)
    data = load_results(args.input)
    recovery_figure(data, output)
    event_zoom_figure(data, output)
    corner_figure(data, output)


if __name__ == "__main__":
    main()
