"""Physical short-event photometry with Table, FFTLog, and VBM on identical data.

All methods use the same finite exposures, observations, priors, Gaussian
likelihood, and table-based affine initialization. Table/emcee is an additional
sampler-control diagnostic; the primary comparison includes both NUTS solvers
and VBM/emcee. Independent brightness integration supplies the injected mean.
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
from scipy.special import log_ndtr, logsumexp

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from example._common import model_solver_metadata, save_figure, save_json, style
from example.inference.code.inference import binned_residuals

recovery = import_module("example.compare-single-vbml.code.experiment")
sample_nuts_until = import_module("example.compare-single-vbml.code.sampling").sample_nuts_until

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
METHOD_COLORS = ("#3D8067", "#247BA0", "#9B1B30")
METHOD_LABELS = ("Table / NUTS", "FFTLog / NUTS", "VBM / emcee")
METHOD_STYLES = ("-", "--", ":")


def physical(theta):
    theta = np.asarray(theta, dtype=float)
    if theta.shape[-1] != len(recovery.LENS_NAMES):
        raise ValueError("Expected four sampled lens coordinates")
    result = np.array(theta, copy=True)
    rho, crossing = np.exp(theta[..., 3]), np.exp(theta[..., 2])
    result[..., 1], result[..., 2], result[..., 3] = (
        np.exp(theta[..., 1]) * rho,
        crossing / rho,
        rho,
    )
    return result


def transformed(values):
    values = np.asarray(values, dtype=float)
    result = np.array(values[..., :4], copy=True)
    result[..., 1] = np.log(values[..., 1] / values[..., 3])
    result[..., 2] = np.log(values[..., 2] * values[..., 3])
    result[..., 3] = np.log(values[..., 3])
    return result


def reconstruct_flux(posterior, lens_values, *, seed):
    """Draw paired conditional fluxes using this backend's own magnification.

    Accept physical lens states of shape (..., 4). Return joint states, timing/
    probability metadata, and conditional mean/covariance with matching axes.
    This operation is outside all sampler and ESS timers.
    """
    start = perf_counter()
    lens_values = np.asarray(lens_values)
    if lens_values.shape[-1] != len(recovery.LENS_NAMES):
        raise ValueError("Expected four physical lens parameters")
    shape = lens_values.shape[:-1]
    theta = transformed(lens_values.reshape(-1, 4))
    conditional = jax.jit(
        jax.vmap(lambda a: recovery.flux_conditional(a, posterior.data.flux, posterior.data.error))
    )
    means, covariances = [], []
    for begin in range(0, len(theta), 16):
        result = conditional(posterior.magnifications(theta[begin : begin + 16]))
        means.append(np.asarray(result.mean))
        covariances.append(np.asarray(result.covariance))
    means, covariances = np.concatenate(means), np.concatenate(covariances)
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError("Nonfinite conditional flux distribution")
    noise = np.random.default_rng(seed).normal(size=means.shape)
    fluxes = means + np.einsum("nij,nj->ni", np.linalg.cholesky(covariances), noise)
    result = np.concatenate((lens_values.reshape(-1, 4), fluxes), axis=-1).reshape(*shape, 6)
    probabilities = {}
    for index, name in enumerate(("f_s", "f_b")):
        logp = log_ndtr(-means[:, index] / np.sqrt(covariances[:, index, index]))
        logmean = float(logsumexp(logp) - np.log(len(logp)))
        probabilities[name] = dict(
            conditional_probability_mean=float(np.exp(logmean)),
            log_conditional_probability_mean=logmean,
            maximum_conditional_probability=float(np.exp(logp.max())),
            realized_negative_fraction=float(np.mean(fluxes[:, index] < 0)),
        )
    info = dict(
        seconds=perf_counter() - start,
        geometry_states=len(theta),
        seed=seed,
        backend=posterior.backend,
        scope="One correlated Gaussian (f_s,f_b) draw per retained geometry state, using that method's own exposure-averaged magnification. Reconstruction and summaries are outside sampler timing. Flux samples have conditional uncertainty but no ESS assigned.",
        negative_flux_probabilities=probabilities,
        probability_scope="Rao-Blackwell Monte Carlo means over retained geometry draws, including conditional variance. Zero can be floating-point underflow; finite chains do not bound unsampled geometry-tail probability.",
    )
    return result, info, means.reshape(*shape, 2), covariances.reshape(*shape, 2, 2)


def add_flux_summary(row, values, reconstruction):
    row["flux_reconstruction"] = reconstruction
    row["flux_reconstruction_seconds"] = reconstruction["seconds"]
    flat = np.asarray(values).reshape(-1, len(recovery.NAMES))
    for index, name in enumerate(recovery.NAMES[4:], start=4):
        sample = flat[:, index]
        row["parameters"][name] = dict(
            mean=float(sample.mean()),
            sd=float(sample.std()),
            median=float(np.median(sample)),
            q05=float(np.percentile(sample, 5)),
            q95=float(np.percentile(sample, 95)),
            scope="Derived paired conditional Gaussian draws; excluded from ESS/stopping",
        )


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
        self.subtimes = np.asarray(recovery.exposure_times(data.times)).reshape(-1)
        self.exposure_weights = recovery.exposure_rule(recovery.EXPOSURE_ORDER)[1]
        if backend == "VBM":
            start = perf_counter()
            import VBMicrolensing

            self.vbm = VBMicrolensing.VBMicrolensing()
            self.vbm.Tol = self.vbm.RelTol = vbm_tolerance
            # Resolve at least two annuli: disk and point magnifications can
            # coincide within tolerance even when limb darkening matters.
            self.vbm.minannuli = minannuli
            self.vbm.a1 = recovery.CONFIG["event"]["a1"]
            if vbm_interface == "array":
                adapter = import_module("example.compare-single-vbml.code.vbm_array")
                self.vbm_array = adapter.VBMArray(tolerance=vbm_tolerance, minannuli=minannuli)
            self.backend_setup_seconds = perf_counter() - start

        def log_prob(theta, magnification):
            return recovery.log_likelihood(
                magnification, data.flux, data.error
            ) + recovery.log_prior(theta)

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
        self.sl_magnification = jax.jit(
            jax.vmap(
                lambda theta: recovery.magnification(model, data.times, recovery.unpack(theta))
            )
        )

    def separations(self, theta):
        p = physical(np.atleast_2d(theta))
        tau = (self.subtimes[None, :] - p[:, 0, None]) / p[:, 2, None]
        beta = np.broadcast_to(p[:, 1, None], tau.shape).copy()
        return np.hypot(tau, beta)

    def magnifications(self, theta):
        """Evaluate identical exposure nodes through the chosen VBM API.

        The array bridge calls the installed ESPLMag2 directly. The stock API
        maps separations to times with u0=0, tE=1, t0=0. Neither loops over
        epochs in Python; only the stock path builds lists and unused positions.
        """
        theta = np.atleast_2d(theta)
        if self.backend == "singlens":
            return np.asarray(self.sl_magnification(theta))
        separations = self.separations(theta)
        if self.vbm_interface == "array":
            values = self.vbm_array.magnifications(
                np.exp(theta[:, 3]),
                np.full(len(theta), recovery.CONFIG["event"]["a1"]),
                separations,
            )
        else:
            values = np.array(
                [
                    self.vbm.ESPLLightCurve([-np.inf, 0.0, 0.0, row[3]], u.tolist())[0]
                    for row, u in zip(theta, separations)
                ]
            )
        return values.reshape(len(theta), len(self.times), -1) @ self.exposure_weights

    def __call__(self, theta):
        theta = np.atleast_2d(theta)
        if self.backend == "singlens":
            result = np.asarray(self.sl_log_prob(theta))
        else:
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                ratio = np.hypot(
                    (
                        recovery.CONFIG["season_days"] / 2
                        + abs(theta[:, 0])
                        + recovery.CONFIG["exposure_seconds"] / 86400
                    )
                    / np.exp(theta[:, 2]),
                    np.exp(theta[:, 1]),
                )
            valid = (
                np.isfinite(theta).all(axis=1)
                & (theta[:, 3] >= np.log(1e-6))
                & (theta[:, 3] <= np.log(10))
                & (ratio <= 1e8)
            )
            result = np.full(len(theta), -np.inf)
            if valid.any():
                result[valid] = np.asarray(
                    self.log_prob(theta[valid], self.magnifications(theta[valid]))
                )
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
        conditional = jax.vmap(
            lambda a: recovery.flux_conditional(a, posterior.data.flux, posterior.data.error)
        )(values)
        flux_error = (
            (values - refined)
            * np.asarray(conditional.mean)[:, 0, None]
            / np.asarray(posterior.data.error)
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
    if np.asarray(chains).ndim != 3 or chains.shape[-1] != len(recovery.LENS_NAMES):
        raise ValueError("ESS requires (step, series, four lens parameters)")
    tau = autocorrelation(chains)
    ess = chains.shape[0] * chains.shape[1] / tau
    values = chains.reshape(-1, len(recovery.LENS_NAMES))
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
            for i, name in enumerate(recovery.LENS_NAMES)
        },
        ess_parameter_names=list(recovery.LENS_NAMES),
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
        ensemble_mean_difference_mcse=dict(zip(recovery.LENS_NAMES, agreement.tolist())),
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
    rows = [row for row in report["results"] if row.get("primary_comparison", True)]
    for i, (row, color) in enumerate(zip(rows, METHOD_COLORS)):
        label = row.get("label", f"{row['backend']} / {row['sampler']}")
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
            [row["parameters"][name]["ess"] / row["total_seconds"] for name in recovery.LENS_NAMES],
            np.arange(len(recovery.LENS_NAMES)) + (i - 1) * 0.2,
            s=22,
            color=color,
            label=label,
        )
    axes[0].set(
        xlabel="Elapsed time including initialization and warmup [s]",
        ylabel="Minimum lens-parameter ESS",
        title="Sampling progress",
    )
    axes[0].axhline(report["target_ess"], color="0.5", ls="--", lw=0.8)
    axes[0].legend(fontsize=11)
    axes[1].set(
        xlabel="ESS / s (total runtime)",
        xlim=(0, None),
        title="Efficiency by parameter",
        yticks=np.arange(len(recovery.LENS_NAMES)),
        yticklabels=recovery.LABELS[:4],
    )
    axes[1].invert_yaxis()
    return fig


def posterior_magnitudes(models, data, draws_by_solver, vbm_posterior, emcee_chains):
    """Pointwise intervals, evaluating each posterior with its own lens backend."""
    samples = [
        *[
            np.column_stack([np.asarray(draws[name]).reshape(-1) for name in recovery.NAMES])
            for draws in (draws_by_solver["table"], draws_by_solver["fft"])
        ],
        emcee_chains.reshape(-1, len(recovery.NAMES)),
    ]
    bands = []
    for backend, values in zip(("table", "fft", "VBM"), samples):
        values = values[np.linspace(0, len(values) - 1, 256, dtype=int)]
        theta = transformed(values)
        if backend != "VBM":
            model = models[backend]
            evaluate = jax.jit(
                jax.vmap(lambda x: recovery.magnification(model, data.times, recovery.unpack(x)))
            )
            # Keep the exposure-node batch bounded when plotting a full season.
            mag = np.concatenate(
                [
                    np.asarray(evaluate(jnp.asarray(theta[start : start + 16])))
                    for start in range(0, len(theta), 16)
                ]
            )
        else:
            mag = vbm_posterior.magnifications(theta)
        flux = values[:, 4, None] * mag + values[:, 5, None]
        if np.any(flux <= 0) or not np.isfinite(flux).all():
            raise ValueError("Posterior flux must be finite and positive.")
        bands.append(
            np.percentile(
                recovery.CONFIG["baseline_magnitude_ab"] - 2.5 * np.log10(flux),
                [5, 50, 95],
                axis=0,
            )
        )
    return bands


def plot_recovery(data, bands):
    """The inference data and three recovered curves, with true-lens residuals."""
    times = np.asarray(data.times) - data.truth["t0"]
    zero = recovery.CONFIG["baseline_magnitude_ab"]
    observed = zero - 2.5 * np.log10(np.asarray(data.flux))
    truth = zero - 2.5 * np.log10(np.asarray(data.lens_flux))
    btime, bvalue, berror = binned_residuals(
        data, width=2 * recovery.CONFIG["cadence_minutes"] / 1440
    )
    btime -= data.truth["t0"]
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 6), sharex="col", height_ratios=[2, 1], layout="constrained"
    )
    peak_window = 4 * data.truth["t_e"]
    for column, (lo, hi) in enumerate(((times.min(), times.max()), (-peak_window, peak_window))):
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
        upper.set_title(
            f"{recovery.CONFIG['season_days']:g}-day season" if column == 0 else "Planetary transit"
        )
    axes[0, 0].set_ylabel("F146 AB [mag]")
    axes[1, 0].set_ylabel("Δmag from true lens")
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=11
    )
    return fig


def plot_posterior_comparison(table_draws, fft_draws, emcee_chains, truth):
    """Overlay actual posterior samples, with common bins and normalized marginals."""
    samples = [
        *[
            np.column_stack([np.asarray(draws[name]).reshape(-1) for name in recovery.NAMES])
            for draws in (table_draws, fft_draws)
        ],
        np.asarray(emcee_chains).reshape(-1, len(recovery.NAMES)),
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
            labelpad=0.2,
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


def plot_trace_comparison(table_draws, fft_draws, emcee_trace, truth):
    """Show chain 0 or ensemble 0/walker 0 on each method's own step axis."""
    samples = [
        *[
            np.column_stack([draws[name][0] for name in recovery.NAMES])
            for draws in (table_draws, fft_draws)
        ],
        np.asarray(emcee_trace),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(14, 10), sharex="col", layout="constrained")
    for column, (values, color, label) in enumerate(zip(samples, METHOD_COLORS, METHOD_LABELS)):
        for row, (name, ylabel) in enumerate(zip(recovery.LENS_NAMES, recovery.LABELS[:4])):
            ax = axes[row, column]
            ax.plot(values[:, row], color=color, alpha=0.65, lw=0.6)
            ax.axhline(truth[name], color="0.3", lw=0.7, ls="--", label="Injected")
            ax.set_ylabel(ylabel)
            ax.grid(False)
        axes[0, column].set_title(label)
        axes[-1, column].set_xlabel("Post-warmup steps")
    axes[0, 0].legend(fontsize=9)
    return fig


def check_reference_magnification(models, data, states, vbm_posterior=None):
    """Independent source integration at selected epochs; excluded from timers."""
    result = {}
    geometry = models["table"]
    for state_name, params in states.items():
        parameters = recovery.event_parameters(params)
        u = np.abs(np.asarray(geometry.trajectory(data.times, parameters)))
        rho, f_s = (float(params[name]) for name in ("rho", "f_s"))
        ratio = u / rho
        indices = np.unique(
            np.r_[
                np.linspace(0, len(u) - 1, 32, dtype=int),
                np.argsort(ratio)[:16],
                np.argsort(np.abs(ratio - 1))[:16],
            ]
        )
        reference = recovery.reference_magnification(
            np.asarray(data.times)[indices], params, order=recovery.EXPOSURE_ORDER, spatial_order=64
        )
        checks = {}
        for name, model in models.items():
            actual = np.asarray(recovery.magnification(model, data.times[indices], params))
            refined = np.asarray(
                recovery.magnification(model, data.times[indices], params, order=9)
            )
            scaled = (actual - reference) * f_s / np.asarray(data.error)[indices]
            checks[name] = dict(
                max_relative_magnification_error=float(np.max(np.abs(actual / reference - 1))),
                max_abs_flux_difference_over_error=float(np.max(np.abs(scaled))),
                rms_flux_difference_over_error=float(np.sqrt(np.mean(scaled**2))),
                exposure_refinement_max_flux_error_sigma=float(
                    np.max(np.abs((actual - refined) * f_s / np.asarray(data.error)[indices]))
                ),
            )
        if vbm_posterior is not None:
            theta = transformed(np.array([[params[name] for name in recovery.NAMES]]))
            actual = vbm_posterior.magnifications(theta)[0, indices]
            scaled = (actual - reference) * f_s / np.asarray(data.error)[indices]
            checks["VBM"] = dict(
                max_relative_magnification_error=float(np.max(np.abs(actual / reference - 1))),
                max_abs_flux_difference_over_error=float(np.max(np.abs(scaled))),
                rms_flux_difference_over_error=float(np.sqrt(np.mean(scaled**2))),
            )
        result[state_name] = dict(
            epochs=len(indices),
            epoch_indices=indices.tolist(),
            parameters={name: float(params[name]) for name in recovery.NAMES},
            solvers=checks,
        )
        if state_name == "common_map":
            times = np.asarray(data.times)
            contacts = [
                candidates[np.argmin(np.abs(ratio[candidates] - 1))]
                for candidates in (
                    np.flatnonzero(times <= float(params["t0"])),
                    np.flatnonzero(times > float(params["t0"])),
                )
            ]
            exposure_values = {
                order: recovery.reference_magnification(
                    times[contacts], params, order=order, spatial_order=64
                )
                for order in (recovery.EXPOSURE_ORDER, 15)
            }
            difference = float(
                np.max(
                    abs(exposure_values[recovery.EXPOSURE_ORDER] - exposure_values[15])
                    * f_s
                    / np.asarray(data.error)[contacts]
                )
            )
            result[state_name]["independent_contact_exposure_refinement"] = dict(
                epoch_indices=[int(index) for index in contacts],
                fit_order=recovery.EXPOSURE_ORDER,
                reference_order=15,
                max_flux_error_sigma=difference,
                threshold_sigma=0.01,
                passed=bool(np.isfinite(difference) and difference <= 0.01),
            )
            if not result[state_name]["independent_contact_exposure_refinement"]["passed"]:
                raise RuntimeError("Independent contact exposure refinement exceeds 0.01 sigma")
    return dict(
        scope="Independent spatial brightness quadrature with the same fitted exposure rule, at selected season/peak/contact epochs; isolates lens-solver error. Exposure refinement is checked separately at the two MAP contacts and across posterior draws. No full-data likelihood inferred from this subset.",
        exposure_order=recovery.EXPOSURE_ORDER,
        reference_tolerance=1e-11,
        states=result,
    )


def check_exposure_refinement(model, data, states_by_method):
    """Validate exposure integration on actual posterior states, outside timers.

    The accurate table isolates time-integration error; independent brightness
    references at truth/MAP/means validate the spatial kernel separately.
    """
    times, error = np.asarray(data.times), np.asarray(data.error)
    rows = []
    for method, states in states_by_method.items():
        selected = states[np.linspace(0, len(states) - 1, min(12, len(states)), dtype=int)]
        for index, values_physical in enumerate(selected):
            params = dict(zip(recovery.NAMES, map(float, values_physical)))
            ratio = np.hypot((times - params["t0"]) / params["t_e"], params["u0"]) / params["rho"]
            epochs = np.unique(np.r_[np.argsort(abs(ratio - 1))[:16], np.argsort(ratio)[:8]])
            values = {
                order: np.asarray(recovery.magnification(model, times[epochs], params, order=order))
                for order in (recovery.EXPOSURE_ORDER, 9, 15)
            }
            scale = abs(params["f_s"]) / error[epochs]
            rows.append(
                dict(
                    method=method,
                    state=index,
                    epochs=len(epochs),
                    parameters=params,
                    fit_vs_15_max_flux_error_sigma=float(
                        np.max(abs(values[recovery.EXPOSURE_ORDER] - values[15]) * scale)
                    ),
                    nine_vs_15_max_flux_error_sigma=float(
                        np.max(abs(values[9] - values[15]) * scale)
                    ),
                )
            )
    worst = max(row["fit_vs_15_max_flux_error_sigma"] for row in rows)
    return dict(
        scope="Up to 12 well-spaced actual draws per method, peak/contact epochs, fitted exposure rule versus 9/15 nodes using the independently checked table kernel; outside timing",
        fit_order=recovery.EXPOSURE_ORDER,
        max_flux_error_sigma=worst,
        threshold_sigma=0.01,
        passed=bool(worst <= 0.01),
        states=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=1000, help="emcee steps per ensemble")
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument(
        "--nuts-samples", type=int, default=800, help="minimum draws per NUTS chain"
    )
    parser.add_argument("--nuts-max-samples", type=int, default=8000)
    parser.add_argument("--target-ess", type=float, default=1000)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=64000)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--data-seed", type=int, default=recovery.DATA_SEED)
    parser.add_argument(
        "--reference-data",
        type=Path,
        help="reuse or create the checksum-verified independent injection NPZ",
    )
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
        or args.walkers < 2 * len(recovery.LENS_NAMES)
        or args.chunk < 100
        or args.max_steps < args.chunk
        or args.max_steps % args.chunk
    ):
        parser.error(
            "Use positive warmup/ESS, >=100 NUTS draws/chunk, enough walkers, and valid sampling limits."
        )
    jax.config.update("jax_enable_x64", True)
    data = recovery.synthetic_data(seed=args.data_seed, reference=args.reference_data)
    jax.block_until_ready((data.times, data.flux, data.error, data.lens_flux))
    models, model_setup_seconds = {}, {}
    for name in ("table", "fft"):
        start = perf_counter()
        models[name] = recovery.make_model(name)
        model_setup_seconds[name] = perf_counter() - start
    model = models["table"]
    # Fail before sampling if the requested interface cannot be built/loaded.
    # This setup is charged once to VBM, separately from common initialization.
    try:
        vbm_posterior = Posterior(model, data, "VBM", vbm_interface=args.vbm_interface)
    except RuntimeError as error:
        parser.error(str(error))
    posteriors = {name: Posterior(item, data, "singlens") for name, item in models.items()}
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
    map_seconds = perf_counter() - start
    init_seconds = model_setup_seconds["table"] + map_seconds
    if not initial.success:
        raise RuntimeError(initial.message)
    experiment = {
        **recovery.experiment_metadata(model, data),
        "fit_solvers": {name: model_solver_metadata(item) for name, item in models.items()},
        "model_setup_seconds": model_setup_seconds,
        "map_seconds_including_compile": map_seconds,
        "common_initialization_seconds": init_seconds,
        "model_setup_scope": "Measured fitting-model construction; table setup is part of common initialization. Independent data generation precedes all fitting models. This is an in-process comparison, not a cold-process startup benchmark.",
    }
    save_json(
        dict(
            **experiment,
            theta=initial.x.tolist(),
            covariance=initial.hess_inv.tolist(),
            elapsed_seconds_including_compile=init_seconds,
            map={k: float(v) for k, v in point.items()},
            marginal_hessian=initial.marginal_hessian.tolist(),
            profile_nfev=int(initial.profile_nfev),
            whitened_gradient_max=initial.whitened_gradient_max,
            optimizer_message=str(initial.message),
            map_success_criterion="Finite marginal objective and max absolute gradient in profile-whitened coordinates <1e-5; actual marginal Hessian positive definite",
        ),
        "inference_data",
        args.output_dir,
    )
    print(
        f"Generated {len(data.times)} independent finite-exposure observations (seed {args.data_seed}); common table MAP complete.",
        flush=True,
    )

    def check_nuts(draws, diagnostics, fields):
        irrecoverable = bool(fields["diverging"].any())
        chains = np.stack([draws[name] for name in recovery.LENS_NAMES], axis=-1).transpose(1, 0, 2)
        try:
            tau = autocorrelation(chains)
        except RuntimeError:
            return dict(min_ess=0.0, done=False, irrecoverable=irrecoverable)
        min_ess = float(chains.shape[0] * 2 / tau.max())
        done = bool(
            min_ess >= args.target_ess
            and chains.shape[0] >= 50 * tau.max()
            and max(item["r_hat"] for item in diagnostics.values()) < 1.01
            and not irrecoverable
        )
        return dict(min_ess=min_ess, done=done, irrecoverable=irrecoverable)

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
        nominal_mcse_fraction_of_posterior_sd=float(args.target_ess**-0.5),
        ess_target_scope=f"Minimum ESS {args.target_ess:g} for each of four lens parameters targets approximately {100 / np.sqrt(args.target_ess):.2g}% of its posterior SD as Monte Carlo mean error. R-hat, divergence and chain-length criteria also apply. Derived conditional fluxes do not enter ESS/stopping.",
        estimated_time_scope="Projected time rescales production cost to the target ESS after fixed setup/warmup; it does not predict a diagnostics-constrained stopping time. In particular, the 50-tau ensemble requirement can force emcee to exceed the ESS target. Main plots use actual measured time and ESS.",
        posterior_comparison_scope="Pooled posterior-SD differences describe shifts between numerical targets. Combined-MCSE differences are a heuristic assuming independent Monte Carlo errors; methods reuse the same RNG seed and their cross-method covariance is unknown, so these are not formal significance tests.",
        primary_methods=["table_nuts", "fft_nuts", "vbm_emcee"],
        diagnostic_methods=["table_emcee"],
        common_initialization="Bundled-table construction, four-dimensional marginal MAP and its actual Hessian; measured cost charged to every method",
        ess_estimator="emcee integrated_time(c=5), four physical lens-parameter ACFs averaged over series; ESS=N/tau",
        scope="Four sampled lens parameters with source/blend flux analytically marginalized under the same normalized Gaussian prior for every method. Same physical transit, independent injection, recorded noise, fixed limb darkening and exposure average; rectilinear trajectory, no GP. Total time includes common model/MAP/Hessian, extra backend setup, JIT, warmup and production. Conditional flux reconstruction is timed separately; IO, plots, diagnostics, validation and injection are excluded. Primary comparison: Table/NUTS, FFTLog/NUTS, VBM/emcee; Table/emcee is a sampler control. VBM Tol=RelTol=1e-3,minannuli2; numerical accuracy differs between methods. Two chains or ensembles must pass ESS and convergence checks. Six-parameter posterior intervals include conditional flux variance; fluxes have no assigned ESS. Results depend on event and seed.",
        results=[],
    )
    draws_by_solver, theta_by_solver, exposure_states = {}, {}, {}
    for index, name in enumerate(("table", "fft")):
        print(f"Starting {METHOD_LABELS[index]}", flush=True)
        fit_model = models[name]
        draws, hmc = sample_nuts_until(
            lambda theta: recovery.negative_log_posterior(
                theta, fit_model, data.times, data.flux, data.error
            ),
            initial,
            recovery.unpack,
            warmup=args.nuts_warmup,
            samples=args.nuts_samples,
            max_samples=args.nuts_max_samples,
            seed=args.seed,
            check=check_nuts,
        )
        run_name = "inference_nuts" if name == "table" else "inference_fft_nuts"
        save_json(
            {**experiment, **hmc, "fit_solver": experiment["fit_solvers"][name]},
            run_name,
            args.output_dir,
        )
        archive_name = "posterior_singlens_nuts" if name == "table" else "posterior_fftlog_nuts"
        np.savez_compressed(args.output_dir / f"{archive_name}.npz", **draws)
        chains = np.stack([draws[key] for key in recovery.LENS_NAMES], axis=-1).transpose(1, 0, 2)
        row = dict(
            method=f"{name}_nuts",
            label=METHOD_LABELS[index],
            primary_comparison=True,
            backend="singlens",
            solver=name,
            sampler="NUTS",
            **efficiency(
                chains,
                hmc["elapsed_seconds_including_compile"],
                initialization_seconds=init_seconds,
                backend_setup_seconds=model_setup_seconds["fft"] if name == "fft" else 0.0,
            ),
            divergences=hmc["divergences"],
            target_accept_prob=hmc["target_accept_prob"],
            max_r_hat=hmc["max_r_hat"],
            warmup_steps_per_chain=hmc["warmup_per_chain"],
            sampling_history=hmc["sampling_history"],
            warmup_seconds_including_compile=hmc["warmup_seconds_including_compile"],
            production_draws=hmc["production_draws"],
            production_leapfrog_steps=hmc["production_leapfrog_steps"],
            diagnostics_pass=bool(hmc["stopping_check"]["done"]),
        )
        report["results"].append(row)
        theta_by_solver[name] = transformed(
            np.stack([draws[key].reshape(-1) for key in recovery.LENS_NAMES], axis=-1)
        )
        save_json(report, "inference_speed", args.output_dir)
        if not row["diagnostics_pass"]:
            raise RuntimeError(f"{row['label']} convergence failed; partial report and draws saved")
        joint, reconstruction, means, covariances = reconstruct_flux(
            posteriors[name], chains.transpose(1, 0, 2), seed=args.seed + 100 + index
        )
        draws_by_solver[name] = {key: joint[..., i] for i, key in enumerate(recovery.NAMES)}
        add_flux_summary(row, joint, reconstruction)
        exposure_states[name] = joint.reshape(-1, 6)
        np.savez_compressed(
            args.output_dir / f"{archive_name}.npz",
            **draws_by_solver[name],
            conditional_flux_mean=means,
            conditional_flux_covariance=covariances,
        )
        save_json(report, "inference_speed", args.output_dir)

    report["vbm_convergence"] = {}
    for index, posterior in enumerate((vbm_posterior, posteriors["table"])):
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
        is_vbm = posterior.backend == "VBM"
        result = {
            **info,
            "method": "vbm_emcee" if is_vbm else "table_emcee",
            "label": "VBM / emcee" if is_vbm else "Table / emcee",
            "solver": "VBM" if is_vbm else "table",
            "primary_comparison": is_vbm,
            **efficiency(
                chains,
                seconds,
                initialization_seconds=init_seconds,
                backend_setup_seconds=posterior.backend_setup_seconds,
            ),
        }
        result["diagnostics_pass"] &= result["min_ess"] >= args.target_ess
        values = chains.reshape(-1, len(recovery.LENS_NAMES))
        if is_vbm:
            result["interface"] = args.vbm_interface
        if is_vbm and result["diagnostics_pass"]:
            indices = np.linspace(0, len(values) - 1, 64, dtype=int)
            report["vbm_convergence"]["vbm_states"] = check_vbm_convergence(
                posterior, transformed(values[indices])
            )
        report["results"].append(result)
        archive_path = args.output_dir / f"posterior_{posterior.backend.lower()}_emcee.npz"
        retained = chains[::10].reshape(-1, 2, args.walkers, 4).transpose(1, 0, 2, 3)
        np.savez_compressed(
            archive_path,
            chains=retained,
            names=np.array(recovery.LENS_NAMES),
            trace=chains[:, 0, :],
        )
        save_json(report, "inference_speed", args.output_dir)
        if not result["diagnostics_pass"]:
            raise RuntimeError(
                f"{result['label']} convergence failed; partial report and draws saved"
            )
        joint, reconstruction, means, covariances = reconstruct_flux(
            posterior, retained, seed=args.seed + 102 + index
        )
        reconstruction["retained_step_stride"] = 10
        add_flux_summary(result, joint, reconstruction)
        exposure_states[result["method"]] = joint.reshape(-1, 6)
        np.savez_compressed(
            archive_path,
            chains=joint,
            axis_order=np.array(["ensemble", "step", "walker", "parameter"]),
            thinning=10,
            names=np.array(recovery.NAMES),
            conditional_flux_mean=means,
            conditional_flux_covariance=covariances,
            trace=chains[:, 0, :],
            trace_names=np.array(recovery.LENS_NAMES),
            trace_ensemble=0,
            trace_walker=0,
        )
        save_json(report, "inference_speed", args.output_dir)

    theta = theta_by_solver["table"]
    selected = theta[np.linspace(0, len(theta) - 1, 64, dtype=int)]
    report["vbm_interface_agreement"] = check_vbm_interface(vbm_posterior, selected)
    report["vbm_convergence"]["table_nuts_states"] = check_vbm_convergence(vbm_posterior, selected)
    fft_states = theta_by_solver["fft"]
    report["vbm_convergence"]["fft_nuts_states"] = check_vbm_convergence(
        vbm_posterior, fft_states[np.linspace(0, len(fft_states) - 1, 64, dtype=int)]
    )
    baseline_values = posteriors["table"](selected)
    report["backend_log_posterior_difference"] = {}
    for name, posterior in (("fft", posteriors["fft"]), ("VBM", vbm_posterior)):
        delta = posterior(selected) - baseline_values
        report["backend_log_posterior_difference"][name] = dict(
            states=len(selected),
            convention=f"{name} minus table on table/NUTS posterior states",
            minimum=float(delta.min()),
            maximum=float(delta.max()),
            range=float(np.ptp(delta)),
        )
    report["injection_discrepancy"] = {
        name: recovery.injection_discrepancy(item, data) for name, item in models.items()
    }
    report["reference_magnification"] = check_reference_magnification(
        models,
        data,
        {
            "injected": data.truth,
            "common_map": point,
            "table_posterior_mean": {
                name: report["results"][0]["parameters"][name]["mean"] for name in recovery.NAMES
            },
        },
        vbm_posterior=vbm_posterior,
    )
    report["exposure_refinement"] = check_exposure_refinement(
        models["table"], data, exposure_states
    )
    save_json(report, "inference_speed", args.output_dir)
    if not report["exposure_refinement"]["passed"]:
        raise RuntimeError(
            "Fixed exposure integration exceeds 0.01 sigma; refine all backends before using the comparison"
        )
    report["posterior_evaluation"] = {
        name: posterior_costs(
            item, data, selected, walkers=args.walkers, vbm_interface=args.vbm_interface
        )
        for name, item in models.items()
    }
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
            if name in recovery.LENS_NAMES
        }
    save_json(report, "inference_speed", args.output_dir)
    keys = (
        "method",
        "label",
        "primary_comparison",
        "backend",
        "solver",
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
    with np.load(args.output_dir / "posterior_vbm_emcee.npz") as saved:
        save_figure(
            plot_posterior_comparison(
                draws_by_solver["table"], draws_by_solver["fft"], saved["chains"], data.truth
            ),
            "posterior_comparison",
            args.output_dir,
        )
        save_figure(
            plot_trace_comparison(
                draws_by_solver["table"], draws_by_solver["fft"], saved["trace"], data.truth
            ),
            "trace_comparison",
            args.output_dir,
        )
        bands = posterior_magnitudes(models, data, draws_by_solver, vbm_posterior, saved["chains"])
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
