"""Parallax inference: singlens/NUTS versus VBM/emcee on independent-noise data.

All methods use the same observations, priors, Gaussian likelihood, parallax
geometry, default FFT settings, and affine initialization.
"""

import argparse
import csv
import hashlib
import sys
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import corner
import emcee
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from example._common import save_figure, save_json, style
from example.inference.code.inference import binned_residuals

recovery = import_module("example.compare-single-vbml.code.experiment")
sample_nuts_until = import_module("example.compare-single-vbml.code.sampling").sample_nuts_until

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
METHOD_COLORS = ("#247BA0", "#9B1B30", "#3D8067")
METHOD_LABELS = ("singlens / NUTS", "VBM / emcee", "singlens / emcee")
METHOD_STYLES = ("-", "--", ":")


def physical(theta):
    result = np.array(theta, copy=True)
    result[..., 1:5] = np.exp(result[..., 1:5])
    result[..., 8] = expit(result[..., 8])
    return result


def transformed(values):
    result = np.array(values, copy=True)
    result[..., 1:5] = np.log(result[..., 1:5])
    result[..., 8] = logit(result[..., 8])
    return result


class Posterior:
    """A common likelihood; only the magnification calculation is exchanged."""

    def __init__(
        self, model, data, backend, *, vbm_tolerance=1e-3, minannuli=2, vbm_interface="array"
    ):
        self.model, self.data, self.backend = model, data, backend
        if vbm_interface not in ("array", "python"):
            raise ValueError("vbm_interface must be 'array' or 'python'")
        self.vbm_interface = vbm_interface
        self.backend_setup_seconds = 0.0
        self.times = np.asarray(data.times)
        self.basis_n = np.asarray(model.parallax.offsets(data.times, pi_e_n=1.0, pi_e_e=0.0))
        self.basis_e = np.asarray(model.parallax.offsets(data.times, pi_e_n=0.0, pi_e_e=1.0))
        if backend == "VBM":
            start = perf_counter()
            import VBMicrolensing

            self.vbm = VBMicrolensing.VBMicrolensing()
            self.vbm.Tol = self.vbm.RelTol = vbm_tolerance
            # Resolve at least two annuli: disk and point magnifications can
            # coincide within tolerance even when limb darkening matters.
            self.vbm.minannuli = minannuli
            self.vbm.a1 = 0.0
            if vbm_interface == "array":
                adapter = import_module("example.compare-single-vbml.code.vbm_array")
                self.vbm_array = adapter.VBMArray(tolerance=vbm_tolerance, minannuli=minannuli)
            self.backend_setup_seconds = perf_counter() - start

        def log_prob(theta, magnification):
            p = recovery.unpack(theta)
            return recovery.log_likelihood(magnification, p, data.flux, data.error) - (
                0.5 * jnp.sum(((theta - recovery.PRIOR_LOC) / recovery.PRIOR_SCALE) ** 2)
            )

        self.log_prob = jax.jit(jax.vmap(log_prob))
        self.sl_log_prob = jax.jit(
            jax.vmap(
                lambda theta: (
                    -recovery.negative_log_posterior(
                        theta, model, data.times, data.flux, data.error
                    )
                )
            )
        )

    def separations(self, theta):
        p = physical(np.atleast_2d(theta))
        tau = (self.times[None, :] - p[:, 0, None]) / p[:, 2, None]
        beta = np.broadcast_to(p[:, 1, None], tau.shape).copy()
        for index, basis in ((6, self.basis_n), (7, self.basis_e)):
            tau += p[:, index, None] * basis[0]
            beta += p[:, index, None] * basis[1]
        return np.hypot(tau, beta)

    def magnifications(self, theta):
        """Evaluate the common parallax separations through the chosen VBM API.

        The array bridge calls the installed ESPLMag2 directly. The stock API
        maps separations to times with u0=0, tE=1, t0=0. Neither loops over
        epochs in Python; only the stock path builds lists and unused positions.
        """
        theta = np.atleast_2d(theta)
        separations = self.separations(theta)
        if self.vbm_interface == "array":
            return self.vbm_array.magnifications(
                np.exp(theta[:, 3]), expit(theta[:, 8]), separations
            )
        curves = []
        for row, u in zip(theta, separations):
            self.vbm.a1 = float(expit(row[8]))
            curves.append(self.vbm.ESPLLightCurve([-np.inf, 0.0, 0.0, row[3]], u.tolist())[0])
        return np.asarray(curves)

    def __call__(self, theta):
        theta = np.atleast_2d(theta)
        if self.backend == "singlens":
            result = np.asarray(self.sl_log_prob(theta))
        else:
            result = np.asarray(self.log_prob(theta, self.magnifications(theta)))
        return np.where(np.isfinite(result), result, -np.inf)


def autocorrelation(chains):
    """Common per-parameter IAT estimator, averaging ACFs over series."""
    tau = emcee.autocorr.integrated_time(chains, tol=0)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise RuntimeError("Invalid autocorrelation estimate; extend the chains.")
    return tau


def check_vbm_convergence(posterior, theta):
    """Check the numerical target at stricter tolerance outside MCMC timing."""
    reference = Posterior(
        posterior.model,
        posterior.data,
        "VBM",
        vbm_tolerance=1e-6,
        minannuli=4,
        vbm_interface=posterior.vbm_interface,
    )
    differences = []
    max_flux_error = 0.0
    for block in np.array_split(theta, (len(theta) + 63) // 64):
        values = posterior.magnifications(block)
        refined = reference.magnifications(block)
        differences.extend(
            np.asarray(posterior.log_prob(block, values))
            - np.asarray(posterior.log_prob(block, refined))
        )
        flux_error = (
            (values - refined) * np.exp(block[:, 4, None]) / np.asarray(posterior.data.error)
        )
        max_flux_error = max(max_flux_error, float(np.max(np.abs(flux_error))))
    differences = np.asarray(differences)
    return dict(
        states=len(theta),
        reference_tolerance=1e-6,
        reference_minannuli=4,
        delta_log_posterior_range=[float(differences.min()), float(differences.max())],
        max_abs_flux_difference_over_error=max_flux_error,
        max_abs_log_posterior_difference=float(np.max(np.abs(differences))),
    )


def efficiency(chains, seconds, *, initialization_seconds, backend_setup_seconds=0.0):
    # chains shape: production steps, series (chains/walkers), physical parameters.
    tau = autocorrelation(chains)
    ess = chains.shape[0] * chains.shape[1] / tau
    values = chains.reshape(-1, len(recovery.NAMES))
    return dict(
        sampling_seconds_including_warmup_compile=seconds,
        initialization_seconds=initialization_seconds,
        backend_setup_seconds=backend_setup_seconds,
        total_seconds=seconds + initialization_seconds + backend_setup_seconds,
        min_ess=float(ess.min()),
        min_ess_per_second=float(
            ess.min() / (seconds + initialization_seconds + backend_setup_seconds)
        ),
        steps_per_series=chains.shape[0],
        series=chains.shape[1],
        length_over_max_tau=float(chains.shape[0] / tau.max()),
        parameters={
            name: dict(
                tau=float(tau[i]),
                ess=float(ess[i]),
                mean=float(values[:, i].mean()),
                sd=float(values[:, i].std()),
                median=float(np.median(values[:, i])),
                q05=float(np.percentile(values[:, i], 5)),
                q95=float(np.percentile(values[:, i], 95)),
                mcse_mean=float(values[:, i].std() / np.sqrt(ess[i])),
            )
            for i, name in enumerate(recovery.NAMES)
        },
    )


def run_emcee(posterior, initial, *, warmup, chunk, max_steps, walkers, seed, output, target_ess):
    center, transform = initial.x, np.linalg.cholesky(initial.hess_inv)
    evaluated_states = 0
    callback_calls = 0

    def log_prob(z):
        nonlocal evaluated_states, callback_calls
        evaluated_states += len(z)
        callback_calls += 1
        return posterior(center + z @ transform.T)

    samplers, states = [], []
    start = perf_counter()
    for ensemble in range(2):
        rng = np.random.RandomState(seed + ensemble)
        sampler = emcee.EnsembleSampler(walkers, len(center), log_prob, vectorize=True)
        positions = 0.1 * rng.normal(size=(walkers, len(center)))
        sampler.random_state = rng.get_state()
        state = sampler.run_mcmc(positions, warmup)
        print(f"{posterior.backend}/emcee: ensemble {ensemble + 1} warmup complete", flush=True)
        sampler.reset()
        samplers.append(sampler)
        states.append(state)
    warmup_seconds = perf_counter() - start
    warmup_evaluations, warmup_calls = evaluated_states, callback_calls
    previous_tau = None
    converged = False
    history = []
    for stop in range(chunk, max_steps + 1, chunk):
        for i, sampler in enumerate(samplers):
            states[i] = sampler.run_mcmc(states[i], chunk)
        # Exclude convergence diagnostics and checkpoint IO from sampling time.
        elapsed = perf_counter() - start
        check_start = perf_counter()
        arrays = [physical(center + s.get_chain() @ transform.T) for s in samplers]
        taus = np.array([autocorrelation(x) for x in arrays])
        relative_change = (
            float(np.max(np.abs(taus / previous_tau - 1))) if previous_tau is not None else None
        )
        converged = bool(
            np.all(stop >= 50 * taus) and relative_change is not None and relative_change < 0.1
        )
        pooled_tau = autocorrelation(np.concatenate(arrays, axis=1))
        means = np.array([x.mean(axis=(0, 1)) for x in arrays])
        variances = np.array([x.var(axis=(0, 1)) for x in arrays])
        ess_per_ensemble = stop * walkers / taus
        difference_mcse = np.sqrt((variances / ess_per_ensemble).sum(axis=0))
        agreement = np.abs(means[0] - means[1]) / difference_mcse
        min_ess = float(stop * 2 * walkers / pooled_tau.max())
        converged &= min_ess >= target_ess and bool(np.max(agreement) < 4)
        history.append(
            dict(
                steps=stop,
                max_tau=float(taus.max()),
                relative_change=relative_change,
                elapsed_seconds=elapsed,
                min_ess=min_ess,
            )
        )
        print(
            f"{posterior.backend}/emcee: {stop} steps/ensemble, max tau={taus.max():.1f}, "
            f"elapsed={elapsed:.1f}s, converged={converged}",
            flush=True,
        )
        # Checkpoints allow an interrupted run to remain inspectable.
        save_json(
            dict(history=history, elapsed_seconds=elapsed), f"progress_{posterior.backend}", output
        )
        previous_tau = taus
        start += perf_counter() - check_start
        if converged:
            break
    chains = np.concatenate(arrays, axis=1)
    info = dict(
        backend=posterior.backend,
        sampler="emcee",
        ensembles=2,
        walkers_per_ensemble=walkers,
        warmup_steps_per_ensemble=warmup,
        warmup_seconds_including_compile=warmup_seconds,
        posterior_evaluations=dict(
            warmup_including_initial=warmup_evaluations,
            production=evaluated_states - warmup_evaluations,
            total=evaluated_states,
            warmup_callback_calls=warmup_calls,
            production_callback_calls=callback_calls - warmup_calls,
        ),
        acceptance_fraction=float(np.mean([s.acceptance_fraction.mean() for s in samplers])),
        autocorrelation_history=history,
        max_ensemble_mean_difference_mcse=float(agreement.max()),
        ensemble_mean_difference_mcse=dict(zip(recovery.NAMES, agreement.tolist())),
        diagnostics_pass=converged and bool(np.max(agreement) < 4),
    )
    return chains, elapsed, info


def check_vbm_interface(posterior, theta):
    """Require the array interface to preserve the stock numerical target."""
    stock = Posterior(
        posterior.model,
        posterior.data,
        "VBM",
        vbm_interface="python",
        vbm_tolerance=posterior.vbm.Tol,
        minannuli=posterior.vbm.minannuli,
    )
    actual, expected = posterior.magnifications(theta), stock.magnifications(theta)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    delta = np.asarray(posterior.log_prob(theta, actual)) - np.asarray(
        stock.log_prob(theta, expected)
    )
    np.testing.assert_allclose(delta, 0.0, rtol=0, atol=1e-8)
    return dict(
        states=len(theta),
        interface=posterior.vbm_interface,
        max_magnification_relative_difference=float(np.max(np.abs(actual / expected - 1))),
        max_log_posterior_difference=float(np.max(np.abs(delta))),
        magnification_rtol=1e-12,
        magnification_atol=1e-12,
        log_posterior_atol=1e-8,
        passed=True,
    )


def posterior_costs(model, data, theta, *, walkers, vbm_interface):
    """Synchronized value/gradient and walker-batch costs, outside MCMC timing."""
    timed = import_module("example.compare-single-vbml.code.benchmark").timed

    def objective(x):
        return recovery.negative_log_posterior(x, model, data.times, data.flux, data.error)

    value = jax.jit(objective)
    value_gradient = jax.jit(jax.value_and_grad(objective))
    backends = {
        "singlens": Posterior(model, data, "singlens"),
        "VBM_python": Posterior(model, data, "VBM", vbm_interface="python"),
    }
    if vbm_interface == "array":
        backends["VBM_array"] = Posterior(model, data, "VBM", vbm_interface="array")
    result = dict(
        scope="Warm full posterior callbacks on evenly spaced NUTS posterior states. "
        "Batch sizes include both stretch-move halves. Input preparation and JIT "
        "excluded from warm medians; every result synchronized. No gradient for VBM.",
        repeats=100,
        singlens_value=timed(lambda: value(theta[0]).block_until_ready(), 100),
        singlens_value_gradient=timed(lambda: jax.block_until_ready(value_gradient(theta[0])), 100),
        batches={},
    )
    rng = np.random.default_rng(0)
    for size in sorted({1, walkers // 2, walkers - walkers // 2}):
        batch = theta[np.linspace(0, len(theta) - 1, size, dtype=int)]
        calls = {name: (lambda p=p: p(batch)) for name, p in backends.items()}
        # Record first calls separately, then interleave warm timings.
        measurements = {name: timed(call, 1) for name, call in calls.items()}
        durations = {name: [] for name in calls}
        for _ in range(100):
            for name in rng.permutation(list(calls)):
                start = perf_counter()
                calls[name]()
                durations[name].append(perf_counter() - start)
        for name, seconds in durations.items():
            measurements[name].update(
                raw_seconds=seconds,
                median_seconds=float(np.median(seconds)),
                p10_seconds=float(np.percentile(seconds, 10)),
                p90_seconds=float(np.percentile(seconds, 90)),
            )
        result["batches"][str(size)] = measurements
    return result


def plot(report):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    rows = report["results"]
    for i, (row, color) in enumerate(zip(rows, METHOD_COLORS)):
        label = f"{row['backend']} / {row['sampler']}"
        history = row.get("sampling_history", row.get("autocorrelation_history"))
        times = [
            h["elapsed_seconds"]
            + row["initialization_seconds"]
            + row.get("backend_setup_seconds", 0.0)
            for h in history
        ]
        ess = [h["min_ess"] for h in history]
        axes[0].plot(times, ess, "o-", ms=3, color=color, label=label)
        axes[0].plot(row["total_seconds"], row["min_ess"], "o", ms=7, color=color)
        axes[1].scatter(
            [row["parameters"][name]["ess"] / row["total_seconds"] for name in recovery.NAMES],
            np.arange(len(recovery.NAMES)) + (i - 1) * 0.2,
            s=22,
            color=color,
            label=label,
        )
    axes[0].set(
        xlabel="Elapsed time including initialization and warmup [s]",
        ylabel="Minimum parameter ESS",
        title="Sampling progress",
    )
    axes[0].axhline(report["target_ess"], color="0.5", ls="--", lw=0.8)
    axes[0].legend(fontsize=11)
    axes[1].set(
        xlabel="ESS / s (total runtime)",
        xlim=(0, None),
        title="Efficiency by parameter",
        yticks=np.arange(len(recovery.NAMES)),
        yticklabels=recovery.LABELS,
    )
    axes[1].invert_yaxis()
    return fig


def posterior_magnitudes(model, data, draws, vbm_posterior, emcee_chains, singlens_chains):
    """Pointwise intervals, evaluating each posterior with its own lens backend."""
    samples = [
        np.column_stack([np.asarray(draws[name]).reshape(-1) for name in recovery.NAMES]),
        emcee_chains.reshape(-1, len(recovery.NAMES)),
        singlens_chains.reshape(-1, len(recovery.NAMES)),
    ]
    bands = []
    for backend, values in zip(("singlens", "VBM", "singlens"), samples):
        values = values[np.linspace(0, len(values) - 1, 256, dtype=int)]
        theta = transformed(values)
        if backend == "singlens":
            evaluate = jax.jit(
                jax.vmap(
                    lambda x: model.magnification(
                        data.times, recovery.event_parameters(recovery.unpack(x))
                    )
                )
            )
            mag = np.asarray(evaluate(jnp.asarray(theta)))
        else:
            mag = vbm_posterior.magnifications(theta)
        flux = values[:, 4, None] * mag + values[:, 5, None]
        if np.any(flux <= 0) or not np.isfinite(flux).all():
            raise ValueError("Posterior flux must be finite and positive.")
        bands.append(np.percentile(25 - 2.5 * np.log10(flux), [5, 50, 95], axis=0))
    return bands


def plot_recovery(data, bands):
    """The inference data and three recovered curves, with true-lens residuals."""
    times = np.asarray(data.times) - data.truth["t0"]
    observed = 25 - 2.5 * np.log10(np.asarray(data.flux))
    truth = 25 - 2.5 * np.log10(np.asarray(data.lens_flux))
    btime, bvalue, berror = binned_residuals(data)
    btime -= data.truth["t0"]
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 6), sharex="col", height_ratios=[2, 1], layout="constrained"
    )
    for column, (lo, hi) in enumerate(((times.min(), times.max()), (-1, 1))):
        upper, lower = axes[:, column]
        upper.plot(times, observed, ".", ms=1.5, color="0.6", alpha=0.3, label="Data")
        upper.plot(times, truth, "--", color="0.25", lw=0.8, label="True lens")
        lower.axhline(0, color="0.25", ls="--", lw=0.8)
        lower.plot(times, observed - truth, ".", ms=1.5, color="0.6", alpha=0.25)
        lower.errorbar(btime, bvalue, berror, fmt=".", ms=2, lw=0.4, color="0.25", alpha=0.6)
        for band, color, label, linestyle in zip(
            bands, METHOD_COLORS, METHOD_LABELS, METHOD_STYLES
        ):
            q05, median, q95 = band
            upper.fill_between(times, q05, q95, color=color, alpha=0.18)
            upper.plot(times, median, color=color, lw=1.2, ls=linestyle, label=label)
            lower.fill_between(times, q05 - truth, q95 - truth, color=color, alpha=0.18)
            lower.plot(times, median - truth, color=color, lw=1.0, ls=linestyle)
        selected = (times >= lo) & (times <= hi)
        lower.set(
            ylim=(
                -1.08 * max(abs(observed[selected] - truth[selected])),
                1.08 * max(abs(observed[selected] - truth[selected])),
            )
        )
        upper.set(
            xlim=(lo, hi), ylim=(max(observed[selected]) + 0.1, min(observed[selected]) - 0.1)
        )
        lower.set(xlabel=r"$t-t_0$ [day]")
        upper.set_title("72-day season" if column == 0 else "Peak ±1 day")
    axes[0, 0].set_ylabel("W149 [mag]")
    axes[1, 0].set_ylabel("Δmag from true lens")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=11
    )
    return fig


def plot_posterior_comparison(draws, emcee_chains, singlens_chains, truth):
    """Overlay actual posterior samples, with common bins and normalized marginals."""
    samples = [
        np.column_stack([np.asarray(draws[name]).reshape(-1) for name in recovery.NAMES]),
        np.asarray(emcee_chains).reshape(-1, len(recovery.NAMES)),
        np.asarray(singlens_chains).reshape(-1, len(recovery.NAMES)),
    ]
    injected = np.array([truth[name] for name in recovery.NAMES])
    quantiles = np.array([np.quantile(values, [0.001, 0.999], axis=0) for values in samples])
    lower = np.minimum(quantiles[:, 0].min(axis=0), injected)
    upper = np.maximum(quantiles[:, 1].max(axis=0), injected)
    margin = 0.03 * (upper - lower)
    ranges = list(zip(lower - margin, upper + margin))
    fig = None
    for i, (values, color, linestyle) in enumerate(zip(samples, METHOD_COLORS, METHOD_STYLES)):
        fig = corner.corner(
            values,
            fig=fig,
            labels=recovery.LABELS,
            range=ranges,
            bins=32,
            weights=np.full(len(values), 1 / len(values)),
            color=color,
            truths=injected if i == 0 else None,
            truth_color="0.3",
            smooth=0.8,
            smooth1d=0.8,
            levels=(0.68, 0.95),
            plot_datapoints=False,
            plot_density=False,
            fill_contours=False,
            no_fill_contours=True,
            max_n_ticks=3,
            label_kwargs={"fontsize": 11},
            hist_kwargs={"linewidth": 1.5, "linestyle": linestyle},
            contour_kwargs={"linewidths": 1.2, "linestyles": linestyle},
        )
    handles = [
        Line2D([], [], color=color, lw=2, ls=linestyle, label=name)
        for color, name, linestyle in zip(METHOD_COLORS, METHOD_LABELS, METHOD_STYLES)
    ]
    handles.append(Line2D([], [], color="0.3", lw=1, label="Injected"))
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.93, 0.95), fontsize=20)
    return fig


def plot_trace_comparison(draws, emcee_trace, truth):
    """Fixed NUTS chain 0 and emcee ensemble 0 / walker 0; no time flattening."""
    blue, burgundy = METHOD_COLORS[:2]
    nuts = np.column_stack([draws[name][0] for name in recovery.NAMES])
    values = (
        (8, truth["a1"], r"$a_1$"),
        (2, truth["t_e"], r"$t_E$ [day]"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True, layout="constrained")
    ratio = (len(emcee_trace) - 1) / (len(nuts) - 1)
    for ax, (column, injected, label) in zip(axes, values):
        ax.plot(
            np.arange(len(nuts)),
            nuts[:, column],
            color=blue,
            alpha=0.3,
            lw=0.6,
            label="singlens / NUTS",
        )
        ax.plot(
            np.arange(len(emcee_trace)) / ratio,
            emcee_trace[:, column],
            color=burgundy,
            lw=0.6,
            alpha=0.75,
            label="VBM / emcee",
        )
        ax.axhline(injected, color="0.3", lw=0.7, ls="--", label="Injected")
        ax.set_ylabel(label)
        ax.grid(False)
    top = axes[0].secondary_xaxis("top", functions=(lambda x: x * ratio, lambda x: x / ratio))
    top.set_xlabel("Post-warmup steps (VBM / emcee)", color=burgundy)
    top.tick_params(axis="x", colors=burgundy)
    axes[-1].set(xlim=(0, len(nuts) - 1), xlabel="Post-warmup steps (singlens / NUTS)")
    axes[-1].xaxis.label.set_color(blue)
    axes[-1].tick_params(axis="x", colors=blue)
    axes[0].legend(loc="upper right", ncol=3, fontsize=9)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1000, help="emcee steps per ensemble")
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument(
        "--nuts-samples", type=int, default=800, help="minimum draws per NUTS chain"
    )
    parser.add_argument("--nuts-max-samples", type=int, default=8000)
    parser.add_argument("--target-ess", type=float, default=5000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=16000)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument(
        "--vbm-interface",
        choices=("array", "python"),
        default="array",
        help="array uses the original VBM binary through a C++17 buffer adapter; "
        "python explicitly selects the stock list API",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if (
        min(args.warmup, args.nuts_warmup) < 1
        or args.nuts_samples < 100
        or args.nuts_max_samples < args.nuts_samples
        or args.target_ess <= 0
        or args.walkers < 2 * len(recovery.NAMES)
        or args.chunk < 100
        or args.max_steps < args.chunk
        or args.max_steps % args.chunk
    ):
        parser.error(
            "Use positive warmup/ESS, >=100 NUTS draws/chunk, enough walkers, and valid sampling limits."
        )
    jax.config.update("jax_enable_x64", True)
    model, data = recovery.synthetic_event()
    # Fail before sampling if the requested interface cannot be built/loaded.
    # This cold setup is charged once to VBM, separately from the common MAP.
    try:
        vbm_posterior = Posterior(model, data, "VBM", vbm_interface=args.vbm_interface)
    except RuntimeError as error:
        parser.error(str(error))
    sl_posterior = Posterior(model, data, "singlens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "inference_event.csv",
        np.column_stack([data.times, data.flux, data.error, data.lens_flux]),
        delimiter=",",
        header="time_day,flux,flux_error,lens_flux",
        comments="",
    )
    start = perf_counter()
    initial, point = recovery.fit_event(model, data.times, data.flux, data.error)
    init_seconds = perf_counter() - start
    if not initial.success:
        raise RuntimeError(initial.message)
    experiment = recovery.experiment_metadata(model, data)
    save_json(
        dict(
            **experiment,
            theta=initial.x.tolist(),
            covariance=initial.hess_inv.tolist(),
            elapsed_seconds_including_compile=init_seconds,
            map={k: float(v) for k, v in point.items()},
        ),
        "inference_data",
        args.output_dir,
    )
    print(
        f"Generated {len(data.times)} independent-noise observations; joint MAP complete.",
        flush=True,
    )

    def check_nuts(draws, diagnostics, fields):
        chains = np.stack([draws[name] for name in recovery.NAMES], axis=-1).transpose(1, 0, 2)
        try:
            tau = autocorrelation(chains)
        except RuntimeError:
            return dict(min_ess=0.0, done=False)
        min_ess = float(chains.shape[0] * 2 / tau.max())
        done = bool(
            min_ess >= args.target_ess
            and chains.shape[0] >= 50 * tau.max()
            and max(item["r_hat"] for item in diagnostics.values()) < 1.01
            and not fields["diverging"].any()
        )
        return dict(min_ess=min_ess, done=done)

    draws, hmc = sample_nuts_until(
        lambda theta: recovery.negative_log_posterior(
            theta, model, data.times, data.flux, data.error
        ),
        initial,
        recovery.unpack,
        warmup=args.nuts_warmup,
        samples=args.nuts_samples,
        max_samples=args.nuts_max_samples,
        seed=args.seed,
        check=check_nuts,
    )
    save_json(dict(**experiment, **hmc), "inference_nuts", args.output_dir)
    np.savez_compressed(args.output_dir / "posterior_singlens_nuts.npz", **draws)
    chains = np.stack([draws[name] for name in recovery.NAMES], axis=-1).transpose(1, 0, 2)
    row = dict(
        backend="singlens",
        sampler="NUTS",
        n_fft=model.fft.n_fft,
        **efficiency(
            chains, hmc["elapsed_seconds_including_compile"], initialization_seconds=init_seconds
        ),
        divergences=hmc["divergences"],
        max_r_hat=hmc["max_r_hat"],
        warmup_steps_per_chain=hmc["warmup_per_chain"],
        sampling_history=hmc["sampling_history"],
        warmup_seconds_including_compile=hmc["warmup_seconds_including_compile"],
        production_draws=hmc["production_draws"],
        production_leapfrog_steps=hmc["production_leapfrog_steps"],
        diagnostics_pass=bool(hmc["stopping_check"]["done"]),
    )
    report = dict(
        **experiment,
        versions_external={name: version(name) for name in ("emcee", "VBMicrolensing")},
        vbm_settings=dict(Tol=1e-3, RelTol=1e-3, minannuli=2),
        vbm_interface=args.vbm_interface,
        vbm_adapter=(vbm_posterior.vbm_array.metadata if args.vbm_interface == "array" else None),
        benchmark_source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.iterdir())
            if p.suffix in (".py", ".cpp")
        },
        seed=args.seed,
        target_ess=args.target_ess,
        common_initialization="Joint singlens MAP and covariance; measured cost charged to every method",
        ess_estimator="emcee integrated_time(c=5), physical parameter ACFs averaged over series; ESS=N/tau",
        scope="Nine event, parallax, limb-darkening and flux parameters on newly generated Roman-like independent-noise data. No GP is evaluated. Total time includes common MAP, backend setup (including cold VBM adapter compilation), JIT, warmup and production; excludes IO, plots, diagnostics and numerical validation. Two NUTS chains or two independent emcee ensembles; all runs require minimum ESS>=target and convergence checks. VBM uses the recorded interface, Tol=RelTol=1e-3 and minannuli=2; stricter VBM validation is outside timing. Numerical lens approximations differ; this is not an equal-error comparison. Single seed; rates are not universal speedups.",
        results=[row],
    )
    save_json(report, "inference_speed", args.output_dir)
    theta_nuts = transformed(
        np.stack([draws[name].reshape(-1) for name in recovery.NAMES], axis=-1)
    )
    report["vbm_convergence"] = {}
    for posterior in (vbm_posterior, sl_posterior):
        chains, seconds, info = run_emcee(
            posterior,
            initial,
            warmup=args.warmup,
            chunk=args.chunk,
            max_steps=args.max_steps,
            walkers=args.walkers,
            seed=args.seed,
            output=args.output_dir,
            target_ess=args.target_ess,
        )
        result = {
            **info,
            **efficiency(
                chains,
                seconds,
                initialization_seconds=init_seconds,
                backend_setup_seconds=posterior.backend_setup_seconds,
            ),
        }
        result["diagnostics_pass"] &= result["min_ess"] >= args.target_ess
        if posterior.backend == "VBM":
            result["interface"] = args.vbm_interface
            values = chains.reshape(-1, len(recovery.NAMES))
            indices = np.linspace(0, len(values) - 1, 64, dtype=int)
            report["vbm_convergence"]["vbm_states"] = check_vbm_convergence(
                posterior, transformed(values[indices])
            )
        report["results"].append(result)
        np.savez_compressed(
            args.output_dir / f"posterior_{posterior.backend.lower()}_emcee.npz",
            chains=chains[::10]
            .reshape(-1, 2, args.walkers, len(recovery.NAMES))
            .transpose(1, 0, 2, 3),
            axis_order=np.array(["ensemble", "step", "walker", "parameter"]),
            thinning=10,
            names=np.array(recovery.NAMES),
            trace=chains[:, 0, :],
            trace_ensemble=0,
            trace_walker=0,
        )
        save_json(report, "inference_speed", args.output_dir)
    theta = theta_nuts[np.linspace(0, len(theta_nuts) - 1, 64, dtype=int)]
    report["vbm_interface_agreement"] = check_vbm_interface(vbm_posterior, theta)
    report["vbm_convergence"]["nuts_states"] = check_vbm_convergence(vbm_posterior, theta_nuts)
    delta = vbm_posterior(theta) - sl_posterior(theta)
    report["backend_log_posterior_difference"] = dict(
        states=len(theta),
        convention="VBM minus singlens",
        minimum=float(delta.min()),
        maximum=float(delta.max()),
        range=float(np.ptp(delta)),
    )
    report["posterior_evaluation"] = posterior_costs(
        model, data, theta, walkers=args.walkers, vbm_interface=args.vbm_interface
    )
    baseline = report["results"][0]["parameters"]
    for row in report["results"]:
        overhead = (
            row["initialization_seconds"]
            + row["backend_setup_seconds"]
            + row["warmup_seconds_including_compile"]
        )
        production = row["total_seconds"] - overhead
        row["estimated_seconds_to_target_ess"] = (
            overhead + production * args.target_ess / row["min_ess"]
        )
        if row is report["results"][0]:
            continue
        row["posterior_mean_shift_pooled_sd"] = {
            name: (p["mean"] - baseline[name]["mean"])
            / np.sqrt((p["sd"] ** 2 + baseline[name]["sd"] ** 2) / 2)
            for name, p in row["parameters"].items()
        }
        row["posterior_mean_difference_mcse"] = {
            name: abs(p["mean"] - baseline[name]["mean"])
            / np.hypot(p["mcse_mean"], baseline[name]["mcse_mean"])
            for name, p in row["parameters"].items()
        }
    save_json(report, "inference_speed", args.output_dir)
    keys = (
        "backend",
        "sampler",
        "interface",
        "backend_setup_seconds",
        "total_seconds",
        "min_ess",
        "min_ess_per_second",
        "estimated_seconds_to_target_ess",
        "diagnostics_pass",
    )
    with (args.output_dir / "inference_speed.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(report["results"])
    style()
    save_figure(plot(report), "inference_speed", args.output_dir)
    with (
        np.load(args.output_dir / "posterior_vbm_emcee.npz") as saved,
        np.load(args.output_dir / "posterior_singlens_emcee.npz") as singlens_saved,
    ):
        save_figure(
            plot_posterior_comparison(draws, saved["chains"], singlens_saved["chains"], data.truth),
            "posterior_comparison",
            args.output_dir,
        )
        save_figure(
            plot_trace_comparison(draws, saved["trace"], data.truth),
            "trace_comparison",
            args.output_dir,
        )
        bands = posterior_magnitudes(
            model, data, draws, vbm_posterior, saved["chains"], singlens_saved["chains"]
        )
        save_figure(plot_recovery(data, bands), "inference_recovery", args.output_dir)
    plt.close("all")
    for backend in ("VBM", "singlens"):
        (args.output_dir / f"progress_{backend}.json").unlink(missing_ok=True)
    if not all(row["diagnostics_pass"] for row in report["results"]):
        raise RuntimeError(
            "A sampling convergence check failed; see the report before using timings."
        )


if __name__ == "__main__":
    main()
