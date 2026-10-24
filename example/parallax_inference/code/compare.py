"""Compare singlens Table/NumPyro NUTS and VBMicrolensing/emcee for annual parallax.

Run with ``JAX_PLATFORMS=cpu python -m example.parallax_inference.code.compare``.
The positive-impact-parameter solution is sampled conditionally; a separate
multistart check records competing signed solutions. No sampler is expected
to traverse disconnected parallax modes.
"""

import argparse
import hashlib
import json
import os
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

if __name__ == "__main__":
    os.environ.setdefault("JAX_NUM_CPU_DEVICES", "16")

import emcee
import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import expit

from example._common import metadata, model_solver_metadata, save_json
from example._cpu import cpu_configuration, cpu_seconds
from example.parallax_inference.code import experiment as exp
from example.parallax_inference.code import posterior as target
from example.parallax_inference.code.fit import audit_modes, fit_event, gradient_check
from example.parallax_inference.code.plots import make_figures
from example.parallax_inference.code.vbm import VBMPosterior, validate_vbm

sample_nuts_until = import_module("example.compare-single-vbml.code.sampling").sample_nuts_until
OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
ROOT = Path(__file__).resolve().parents[3]


def autocorrelation(chains):
    """Use the same unthinned, per-series ACF estimator for both samplers."""
    tau = emcee.autocorr.integrated_time(chains, c=5, tol=0)
    if np.any(~np.isfinite(tau)) or np.any(tau <= 0):
        raise RuntimeError("Invalid autocorrelation estimate; more draws are needed")
    return tau


def physical(theta):
    theta = np.asarray(theta)
    result = np.array(theta, copy=True)
    rho = np.exp(theta[..., 3])
    result[..., 1] = rho * np.exp(theta[..., 1])
    result[..., 2] = np.exp(theta[..., 2]) / rho
    result[..., 3] = rho
    result[..., 6] = expit(theta[..., 6])
    result[..., 7:] = exp.FLUX_PRIOR_MEAN + theta[..., 7:] @ target.FLUX_CHOLESKY.T
    return result


def transformed(values):
    values = np.asarray(values)
    result = np.array(values, copy=True)
    result[..., 1] = np.log(abs(values[..., 1]) / values[..., 3])
    result[..., 2] = np.log(values[..., 2] * values[..., 3])
    result[..., 3] = np.log(values[..., 3])
    result[..., 6] = np.log(values[..., 6]) - np.log1p(-values[..., 6])
    result[..., 7:] = np.linalg.solve(
        target.FLUX_CHOLESKY, (values[..., 7:] - exp.FLUX_PRIOR_MEAN)[..., None]
    )[..., 0]
    return result


def efficiency(chains, seconds, initialization, warmup):
    """Chains have axes (step, series, physical lens parameter)."""
    tau = autocorrelation(chains)
    ess = chains.shape[0] * chains.shape[1] / tau
    flat = chains.reshape(-1, len(target.NAMES))
    total = seconds + initialization
    return dict(
        total_seconds=total,
        initialization_seconds=initialization,
        sampling_seconds_including_warmup_compile=seconds,
        warmup_seconds_including_compile=warmup,
        production_seconds=seconds - warmup,
        min_ess=float(ess.min()),
        min_ess_per_second=float(ess.min() / total),
        min_ess_per_production_second=float(ess.min() / (seconds - warmup)),
        steps_per_series=chains.shape[0],
        series=chains.shape[1],
        length_over_max_tau=float(chains.shape[0] / tau.max()),
        parameters={
            name: dict(
                mean=float(flat[:, i].mean()),
                sd=float(flat[:, i].std()),
                q05=float(np.quantile(flat[:, i], 0.05)),
                q95=float(np.quantile(flat[:, i], 0.95)),
                ess=float(ess[i]),
                tau=float(tau[i]),
                mcse_mean=float(flat[:, i].std() / np.sqrt(ess[i])),
            )
            for i, name in enumerate(target.NAMES)
        },
    )


def run_nuts(objective, initial, args):
    def check(draws, diagnostics, fields):
        chains = np.stack([draws[name] for name in target.NAMES], axis=-1).transpose(1, 0, 2)
        irrecoverable = bool(fields["diverging"].any())
        try:
            tau = autocorrelation(chains)
        except RuntimeError:
            return dict(min_ess=0.0, done=False, irrecoverable=irrecoverable)
        minimum = float(chains.shape[0] * chains.shape[1] / tau.max())
        done = bool(
            minimum >= args.target_ess
            and chains.shape[0] >= 50 * tau.max()
            and max(item["r_hat"] for item in diagnostics.values()) < 1.01
            and not fields["diverging"].any()
            and not (fields["num_steps"] >= 1023).any()
        )
        return dict(min_ess=minimum, done=done, irrecoverable=irrecoverable)

    draws, details = sample_nuts_until(
        objective,
        initial,
        target.unpack,
        warmup=args.nuts_warmup,
        samples=args.nuts_samples,
        seed=args.seed,
        max_samples=args.nuts_max_samples,
        check=check,
        chains=args.nuts_chains,
        chain_method=args.nuts_chain_method,
    )
    array = np.stack([draws[name] for name in target.NAMES], axis=-1)
    row = dict(
        method="nuts",
        label="NUTS",
        sampler="NUTS",
        chains=args.nuts_chains,
        chain_method=args.nuts_chain_method,
        warmup_steps_per_chain=args.nuts_warmup,
        diagnostics_pass=bool(details["stopping_check"]["done"]),
        max_r_hat=details["max_r_hat"],
        divergences=details["divergences"],
        tree_depth_saturations=details["tree_depth_saturations"],
        history=details["sampling_history"],
    )
    return array, details["elapsed_seconds_including_compile"], row, details


def run_emcee(objective, initial, args, *, batch_log_prob=None):
    center = jnp.asarray(initial.x)
    transform = jnp.asarray(np.linalg.cholesky(initial.hess_inv))
    if batch_log_prob is None:
        evaluate = jax.jit(jax.vmap(lambda z: -objective(center + transform @ z)))
    else:
        numpy_center, numpy_transform = np.asarray(center), np.asarray(transform)

        def evaluate(z):
            return batch_log_prob(numpy_center + np.asarray(z) @ numpy_transform.T)

    evaluations = 0

    def log_prob(z):
        nonlocal evaluations
        evaluations += len(z)
        return np.asarray(evaluate(z))

    samplers, states = [], []
    start, cpu_start = perf_counter(), cpu_seconds()
    if batch_log_prob is not None:
        batch_log_prob.start_workers(args.emcee_workers)

    def current_cpu():
        workers = 0 if batch_log_prob is None else batch_log_prob.active_worker_cpu_seconds
        return cpu_seconds() + workers

    for ensemble in range(2):
        rng = np.random.RandomState(args.seed + ensemble)
        sampler = emcee.EnsembleSampler(args.walkers, len(target.NAMES), log_prob, vectorize=True)
        positions = 0.1 * rng.normal(size=(args.walkers, len(target.NAMES)))
        sampler.random_state = rng.get_state()
        states.append(sampler.run_mcmc(positions, args.emcee_warmup))
        sampler.reset()
        samplers.append(sampler)
        print(f"emcee: ensemble {ensemble + 1} warmup complete", flush=True)
    warmup = perf_counter() - start
    warmup_cpu = current_cpu() - cpu_start
    previous_tau, converged, history = None, False, []
    for stop in range(args.chunk, args.max_steps + 1, args.chunk):
        for i, sampler in enumerate(samplers):
            states[i] = sampler.run_mcmc(states[i], args.chunk)
        elapsed = perf_counter() - start
        elapsed_cpu = current_cpu() - cpu_start
        diagnostics_start = perf_counter()
        diagnostics_cpu = current_cpu()
        arrays = [
            physical(initial.x + sampler.get_chain() @ np.asarray(transform).T)
            for sampler in samplers
        ]
        taus = np.array([autocorrelation(array) for array in arrays])
        pooled = np.concatenate(arrays, axis=1)
        tau = autocorrelation(pooled)
        relative = None if previous_tau is None else float(np.max(abs(taus / previous_tau - 1)))
        minimum = float(pooled.shape[0] * pooled.shape[1] / tau.max())
        means = np.array([array.mean(axis=(0, 1)) for array in arrays])
        variances = np.array([array.var(axis=(0, 1)) for array in arrays])
        ensemble_ess = stop * args.walkers / taus
        agreement = abs(means[0] - means[1]) / np.sqrt((variances / ensemble_ess).sum(axis=0))
        converged = bool(
            minimum >= args.target_ess
            and np.all(stop >= 50 * taus)
            and relative is not None
            and relative < 0.1
            and agreement.max() < 4
        )
        history.append(
            dict(
                steps=stop,
                elapsed_seconds=elapsed,
                elapsed_cpu_seconds=elapsed_cpu,
                min_ess=minimum,
                max_tau=float(taus.max()),
                relative_tau_change=relative,
                max_ensemble_mean_difference_mcse=float(agreement.max()),
            )
        )
        save_json(dict(history=history), "progress_emcee", args.output_dir)
        print(f"emcee: {stop} steps, ESS={minimum:.0f}, elapsed={elapsed:.1f}s", flush=True)
        start += perf_counter() - diagnostics_start
        cpu_start += current_cpu() - diagnostics_cpu
        previous_tau = taus
        if converged:
            break
    if batch_log_prob is not None:
        batch_log_prob.stop_workers()
    elapsed, elapsed_cpu = perf_counter() - start, cpu_seconds() - cpu_start
    history[-1].update(elapsed_seconds=elapsed, elapsed_cpu_seconds=elapsed_cpu)
    return (
        np.stack(arrays),
        elapsed,
        dict(
            method="emcee",
            label="emcee",
            sampler="emcee",
            ensembles=2,
            walkers_per_ensemble=args.walkers,
            workers=args.emcee_workers,
            warmup_steps_per_ensemble=args.emcee_warmup,
            warmup_seconds_including_compile=warmup,
            diagnostics_pass=converged,
            max_ensemble_mean_difference_mcse=float(agreement.max()),
            posterior_evaluations=evaluations,
            history=history,
        ),
        dict(
            cpu_seconds_including_compile=elapsed_cpu,
            warmup_cpu_seconds_including_compile=warmup_cpu,
        ),
    )


def plot_samples(values):
    """Select actual joint draws for plots; ESS and downloads retain full chains."""
    flat = np.asarray(values).reshape(-1, len(target.NAMES))
    selected = flat[np.linspace(0, len(flat) - 1, min(6000, len(flat)), dtype=int)]
    return dict(zip(target.NAMES, selected.T))


def source_hashes():
    paths = list(Path(__file__).parent.glob("*.py"))
    shared = ROOT / "example/compare-single-vbml/code"
    paths.extend(shared / name for name in ("sampling.py", "vbm_array.py", "vbm_array.cpp"))
    paths.append(ROOT / "example/_cpu.py")
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--data-seed", type=int, default=43)
    parser.add_argument("--seed", type=int, default=18)
    parser.add_argument("--target-ess", type=float, default=5000)
    parser.add_argument("--nuts-warmup", type=int, default=500)
    parser.add_argument("--nuts-samples", type=int, default=1000)
    parser.add_argument("--nuts-max-samples", type=int, default=12000)
    parser.add_argument("--nuts-chains", type=int, default=16)
    parser.add_argument(
        "--nuts-chain-method",
        choices=("sequential", "parallel", "vectorized"),
        default="parallel",
    )
    parser.add_argument("--emcee-warmup", type=int, default=1000)
    parser.add_argument("--walkers", type=int, default=32)
    parser.add_argument("--emcee-workers", type=int, default=16)
    parser.add_argument("--chunk", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=32000)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if (
        args.target_ess <= 0
        or args.nuts_chains < 2
        or args.emcee_workers < 1
        or args.walkers < 2 * len(target.NAMES)
        or min(args.nuts_warmup, args.emcee_warmup) < 1
        or args.chunk < 100
        or args.max_steps < args.chunk
        or args.max_steps % args.chunk
        or not 100 <= args.nuts_samples <= args.nuts_max_samples
    ):
        parser.error("Require positive warmup/ESS, valid sampling lengths, and at least 2d walkers")
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        parser.error("Use JAX_PLATFORMS=cpu for the recorded comparison")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = exp.synthetic_data(seed=args.data_seed, reference=args.reference_data)
    jax.block_until_ready((data.times, data.flux, data.error, data.geometry))
    snapshot = source_hashes()
    start, cpu_start = perf_counter(), cpu_seconds()
    model = exp.make_model("table")
    initial = fit_event(model, data, sign=1)
    initialization = perf_counter() - start
    initialization_cpu = cpu_seconds() - cpu_start
    if not initial.success:
        raise RuntimeError(f"Common MAP failed: {initial.message}")
    vbm = VBMPosterior(data)
    report = dict(
        **metadata(),
        cpu_configuration=cpu_configuration(),
        versions_external={name: version(name) for name in ("emcee", "scipy", "VBMicrolensing")},
        benchmark_source_sha256=snapshot,
        data_seed=args.data_seed,
        sampler_seed=args.seed,
        settings={
            name: value
            for name, value in vars(args).items()
            if name not in ("output_dir", "reference_data", "data_seed", "validate_only")
        },
        data_sha256=exp.data_hash(data),
        reference_data_sha256=exp.reference_data_hash(
            data.times, data.exposure_seconds, data.error, data.lens_flux
        ),
        reference_provenance=exp.reference_provenance(),
        physical_configuration=exp.CONFIG,
        injected=data.truth,
        observations=int(data.times.size),
        lens_parameters=list(exp.LENS_NAMES),
        sampled_parameters=list(target.NAMES),
        flux_treatment="joint; source and blend fluxes sampled, no marginalization",
        target_ess=args.target_ess,
        nominal_mcse_fraction_of_posterior_sd=1 / np.sqrt(args.target_ess),
        ess_estimator="emcee.autocorr.integrated_time(c=5, tol=0) on unthinned physical-parameter series; ESS=N/tau. Walkers in each ensemble are correlated; independent-ensemble agreement is checked separately.",
        branch="u0 > 0; conditional sampling of one parallax solution",
        fit_solvers={"table": model_solver_metadata(model), "vbm": vbm.metadata},
        vbm_settings=dict(tolerance=1e-3, relative_tolerance=1e-3, minannuli=2),
        priors=dict(
            coordinates=list(target.TRANSFORM_NAMES),
            location=target.PRIOR_LOC.tolist(),
            scale=target.PRIOR_SCALE.tolist(),
        ),
        flux_prior=dict(
            mean=exp.FLUX_PRIOR_MEAN.tolist(), covariance=exp.FLUX_PRIOR_COVARIANCE.tolist()
        ),
        reference_checks=data.reference_checks,
        initialization=dict(
            seconds=initialization,
            cpu_seconds=initialization_cpu,
            theta=initial.x.tolist(),
            covariance=initial.hess_inv.tolist(),
            negative_log_posterior=float(initial.fun),
        ),
        scope="singlens Table/NumPyro NUTS versus VBMicrolensing/emcee. Same observations, jointly sampled Gaussian-prior source/blend fluxes, priors, exposure nodes, and Table MAP/Hessian affine coordinates. Numerical lens approximations differ. This compares complete inference implementations, changing both finite-source solver and sampler. Total time includes common model/MAP/Hessian initialization, each backend setup (including the VBM array bridge), compilation, warmup, and sampling. Independent injection, diagnostic checks, signed-mode audit, posterior selection, IO, and figures are excluded and are not inferred from kernel timings. Sampling is conditional on u0>0; no claim of global traversal between parallax modes.",
        results=[],
    )
    np.savetxt(
        args.output_dir / "data.csv",
        np.column_stack((data.times, data.flux, data.error, data.lens_flux, data.exposure_seconds)),
        delimiter=",",
        header="time_day,flux,error,mean,exposure_seconds",
        comments="",
    )
    save_json(report, "comparison", args.output_dir)

    def objective(theta):
        return target.negative_log_posterior(theta, model, data)

    def validate(states):
        mode_audit, positive, _ = audit_modes(model, data)
        difference = None if positive is None else float(initial.fun - positive.fun)
        mode_audit["timed_positive_map_minus_best_nlp"] = difference
        mode_audit["timed_map_verified"] = difference is not None and abs(difference) < 1e-3
        gradient = gradient_check(model, data, initial)
        numerical = {
            "table": exp.validate_model(model, data, states),
            "vbm": validate_vbm(vbm, data, states),
        }
        report.update(
            mode_audit=mode_audit, gradient_check=gradient, numerical_validation=numerical
        )
        return bool(
            mode_audit["both_signs_fitted"]
            and mode_audit["timed_map_verified"]
            and gradient["passed"]
            and all(item["passed"] for item in numerical.values())
        )

    if args.validate_only:
        passed = validate([data.truth, target.unpack(initial.x)])
        report["validation_passed"] = passed
        save_json(report, "validation", args.output_dir)
        if not passed:
            raise RuntimeError("Pre-sampling numerical or mode validation failed")
        return
    draws = {}
    # Alternate order across noise realizations; both methods include their own JIT cost.
    order = ("nuts", "emcee") if args.data_seed % 2 else ("emcee", "nuts")
    report["execution_order"] = list(order)
    for name in order:
        if name == "nuts":
            array, seconds, row, details = run_nuts(objective, initial, args)
        else:
            try:
                array, seconds, row, details = run_emcee(
                    objective, initial, args, batch_log_prob=vbm
                )
            finally:
                vbm.stop_workers()
        row.update(
            solver="table" if name == "nuts" else "vbm",
            backend="singlens" if name == "nuts" else "VBMicrolensing",
            label="singlens Table + NumPyro NUTS" if name == "nuts" else "VBMicrolensing + emcee",
            sampler_library="numpyro" if name == "nuts" else "emcee",
            backend_setup_seconds=0.0 if name == "nuts" else vbm.setup_seconds,
            backend_setup_cpu_seconds=0.0 if name == "nuts" else vbm.setup_cpu_seconds,
        )
        if name == "nuts":
            chains = array.transpose(1, 0, 2)
            warmup = details["warmup_seconds_including_compile"]
            save_json(details, "nuts_diagnostics", args.output_dir)
        else:
            chains = np.concatenate(list(array), axis=1)
            warmup = row["warmup_seconds_including_compile"]
        setup = initialization + row["backend_setup_seconds"]
        row.update(efficiency(chains, seconds, setup, warmup))
        setup_cpu = initialization_cpu + row["backend_setup_cpu_seconds"]
        total_cpu = setup_cpu + details["cpu_seconds_including_compile"]
        row.update(
            total_cpu_seconds=total_cpu,
            initialization_cpu_seconds=setup_cpu,
            sampling_cpu_seconds_including_warmup_compile=details["cpu_seconds_including_compile"],
            warmup_cpu_seconds_including_compile=details["warmup_cpu_seconds_including_compile"],
            production_cpu_seconds=details["cpu_seconds_including_compile"]
            - details["warmup_cpu_seconds_including_compile"],
            average_cpu_utilization=total_cpu / row["total_seconds"],
            min_ess_per_cpu_second=row["min_ess"] / total_cpu,
        )
        row["common_initialization_seconds"] = initialization
        row["history"] = [
            {
                **item,
                "elapsed_seconds": item["elapsed_seconds"] + setup,
                "elapsed_cpu_seconds": item["elapsed_cpu_seconds"] + setup_cpu,
            }
            for item in row["history"]
        ]
        np.savez_compressed(
            args.output_dir / f"{name}_chains.npz",
            values=array,
            names=np.array(target.NAMES),
            axes=np.array(
                ["chain", "draw", "parameter"]
                if name == "nuts"
                else ["ensemble", "step", "walker", "parameter"]
            ),
        )
        report["results"].append(row)
        save_json(report, "comparison", args.output_dir)
        if not row["diagnostics_pass"]:
            raise RuntimeError(f"{name} did not meet the common convergence criteria")
        draws[name] = plot_samples(array)
        np.savez_compressed(args.output_dir / f"{name}_posterior.npz", **draws[name])
    report["results"].sort(key=lambda row: row["method"] != "nuts")
    baseline, comparison = (row["parameters"] for row in report["results"])
    report["posterior_agreement"] = {
        name: dict(
            pooled_sd=abs(baseline[name]["mean"] - comparison[name]["mean"])
            / np.sqrt((baseline[name]["sd"] ** 2 + comparison[name]["sd"] ** 2) / 2),
            combined_mcse=abs(baseline[name]["mean"] - comparison[name]["mean"])
            / np.hypot(baseline[name]["mcse_mean"], comparison[name]["mcse_mean"]),
        )
        for name in target.NAMES
    }
    report["posterior_means_within_4_combined_mcse"] = all(
        item["combined_mcse"] < 4 for item in report["posterior_agreement"].values()
    )
    report["posterior_agreement_scope"] = (
        "Mean differences scaled by pooled posterior SD and combined estimated MCSE; "
        "describes numerical-target differences as well as Monte Carlo error; not a sampler "
        "convergence gate or calibrated significance test. Both algorithms "
        "use the same integer seed but different random-number streams."
    )
    audit_start = perf_counter()
    states = [data.truth, target.unpack(initial.x)]
    for posterior in draws.values():
        for index in np.linspace(0, len(posterior["rho"]) - 1, 6, dtype=int):
            states.append({name: float(posterior[name][index]) for name in exp.NAMES})
    report["validation_passed"] = validate(states)
    report["validation_seconds_excluded"] = perf_counter() - audit_start
    if source_hashes() != snapshot:
        raise RuntimeError("Benchmark source changed during the run")
    for name, checksum in report["source_sha256"].items():
        path = ROOT / "src/singlens" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
            raise RuntimeError(f"Production source changed during the run: {name}")
    report["diagnostics_pass"] = report["validation_passed"] and all(
        row["diagnostics_pass"] for row in report["results"]
    )
    save_json(report, "comparison", args.output_dir)
    if not report["diagnostics_pass"]:
        raise RuntimeError("Numerical, mode, or sampler checks failed; inspect the saved report")
    make_figures(data, report, draws, args.output_dir)
    (args.output_dir / "progress_emcee.json").unlink(missing_ok=True)
    print(
        json.dumps({row["method"]: row["min_ess_per_second"] for row in report["results"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
