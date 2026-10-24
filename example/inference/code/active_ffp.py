"""Roman-like photometry of an active K dwarf magnified by an isolated planet.

The synthetic source and detector scales live in active_settings.py. The
temporal GP describes unresolved rotational modulation, not a stellar surface
map. Its covariance is magnified together with the source flux.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from time import monotonic

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import summary as chain_summary
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize
from scipy.special import logit
from tinygp import GaussianProcess, kernels

import singlens as sl
from example.inference.code.active_checks import physical_parallax_check
from example.inference.code.active_coordinates import (
    SAMPLING_COORDINATES,
    log_prior_jacobian,
    to_prior,
    to_sampling,
)
from example.inference.code.active_settings import configuration, white_noise

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "active_ffp"
NAMES = ("t0", "u0", "t_e", "baseline", "source_fraction", "gp_rms", "gp_scale", "gp_period", "rho")
COORDINATES = (
    "t0",
    "log_u0",
    "log_t_e",
    "log_baseline",
    "logit_source_fraction",
    "log_gp_rms",
    "log_gp_scale",
    "log_gp_period",
    "log_rho",
)
HARMONIC_POWER = 0.2


def build_gp(times, error, params):
    """Two damped rotational harmonics; gp_rms is fractional source-flux RMS.

    gp_period is the fundamental damped period and gp_scale the common
    exponential coherence time. The first harmonic carries 20% of the variance.
    """
    fs = params["baseline"] * params["source_fraction"]
    sigma = fs * params["gp_rms"]
    decay = 1 / params["gp_scale"]
    components = []
    for harmonic, fraction in ((1, 1 - HARMONIC_POWER), (2, HARMONIC_POWER)):
        omega = jnp.hypot(2 * jnp.pi * harmonic / params["gp_period"], decay)
        components.append(
            kernels.quasisep.SHO(
                omega=omega, quality=omega / (2 * decay), sigma=sigma * jnp.sqrt(fraction)
            )
        )
    return GaussianProcess(components[0] + components[1], times, diag=error**2)


# BEGIN ACTIVE GP LIKELIHOOD
def gp_log_probability(magnification, params, times, flux, error):
    """Normalized density for F=A*(fs+g)+fb in observed flux units."""
    fs = params["baseline"] * params["source_fraction"]
    fb = params["baseline"] - fs
    residual = (flux - fb) / magnification - fs
    gp = build_gp(times, error / magnification, params)
    return gp.log_probability(residual) - jnp.log(magnification).sum()


# END ACTIVE GP LIKELIHOOD


class ActiveFFPFit:
    """Nine jointly inferred lens, finite-source, blend, and GP parameters."""

    def __init__(self, seed=20261023, *, times=None):
        self.seed = seed
        self.settings = configuration()
        s = self.settings
        self.times = jnp.asarray(
            np.arange(-s["season_days"] / 2, s["season_days"] / 2, s["cadence_minutes"] / 1440)
            if times is None
            else times
        )
        self.model = sl.Model(source="linear", solver="fft")
        self.rho, self.a1 = s["rho"], 0.4
        nodes, weights = np.polynomial.legendre.leggauss(3)
        self.exposure_nodes = jnp.asarray(nodes * s["exposure_seconds"] / (2 * 86400))
        self.exposure_weights = jnp.asarray(weights / 2)
        self.truth = dict(
            t0=0.2,
            u0=s["u0"],
            t_e=s["t_e_days"],
            baseline=1.0,
            source_fraction=s["source_fraction"],
            gp_rms=s["gp_rms"],
            gp_scale=s["gp_coherence_days"],
            gp_period=s["gp_period_days"],
            rho=self.rho,
        )
        # Independent normal priors in these named transformed coordinates.
        # These are generic event/rotation scales, not the injection values.
        self.prior_mean = np.array(
            [
                0,
                np.log(0.4),
                np.log(0.1),
                0,
                logit(0.75),
                np.log(0.01),
                np.log(15),
                np.log(6),
                np.log(0.6),
            ]
        )
        self.prior_sd = np.array([0.2, 0.7, 0.7, 0.05, 1.2, 1.0, 0.8, 0.7, 0.8])
        self.q_truth = self.pack(self.truth)
        amp = self.magnification(self.truth, self.times)
        fs = self.truth["baseline"] * self.truth["source_fraction"]
        self.lens_flux = fs * amp + 1 - fs
        gp_key, noise_key = jax.random.split(jax.random.key(seed))
        intrinsic = build_gp(self.times, jnp.zeros_like(self.times), self.truth).sample(gp_key)
        self.gp_signal = amp * intrinsic
        latent = self.lens_flux + self.gp_signal
        self.error = jnp.asarray(white_noise(np.asarray(latent), s))
        self.flux = latent + self.error * jax.random.normal(noise_key, self.times.shape)
        self.peak_mag_ab = float(s["baseline_mag_ab"] - 2.5 * jnp.log10(jnp.max(latent)))
        if not np.isfinite(np.asarray(latent)).all() or np.min(latent) <= 0:
            raise ValueError("Synthetic latent flux is not finite and positive")
        if self.peak_mag_ab < s["saturation_guard_mag"]:
            raise ValueError("Lensing + variability + blend violates the saturation guard")
        self.objective = jax.jit(
            jax.value_and_grad(lambda q: -self.loglike(q) + 0.5 * jnp.sum(q * q))
        )

    def pack(self, params):
        values = np.array(
            [
                params["t0"],
                np.log(params["u0"]),
                np.log(params["t_e"]),
                np.log(params["baseline"]),
                logit(params["source_fraction"]),
                np.log(params["gp_rms"]),
                np.log(params["gp_scale"]),
                np.log(params["gp_period"]),
                np.log(params["rho"]),
            ]
        )
        return (values - self.prior_mean) / self.prior_sd

    def physical(self, q):
        p = jnp.asarray(self.prior_mean) + jnp.asarray(q) * jnp.asarray(self.prior_sd)
        return dict(
            zip(
                NAMES,
                [
                    p[..., 0],
                    jnp.exp(p[..., 1]),
                    jnp.exp(p[..., 2]),
                    jnp.exp(p[..., 3]),
                    jax.nn.sigmoid(p[..., 4]),
                    *[jnp.exp(p[..., k]) for k in (5, 6, 7, 8)],
                ],
            )
        )

    def magnification(self, params, times):
        exposure_times = jnp.asarray(times)[:, None] + self.exposure_nodes
        event = sl.Parameters(
            t0=params["t0"], u0=params["u0"], t_e=params["t_e"], rho=params["rho"], a1=self.a1
        )
        amp = self.model.magnification(exposure_times.ravel(), event)
        return amp.reshape(exposure_times.shape) @ self.exposure_weights

    def loglike(self, q):
        p = self.physical(q)
        return gp_log_probability(
            self.magnification(p, self.times), p, self.times, self.flux, self.error
        )

    def fit_map(self, starts=3):
        results, records = [], []
        for k in range(starts):
            initial = np.zeros(len(NAMES))
            initial[0] = float(self.times[jnp.argmax(self.flux)]) / self.prior_sd[0]
            initial[7] = (np.log([4.0, 7.0, 10.0][k % 3]) - self.prior_mean[7]) / self.prior_sd[7]
            result = minimize(
                self.objective,
                initial,
                jac=True,
                method="L-BFGS-B",
                bounds=[(-8, 8)] * len(NAMES),
                options=dict(maxiter=2000, ftol=1e-12, gtol=1e-5, maxls=50),
            )
            finite = bool(
                np.isfinite(result.fun)
                and np.isfinite(result.x).all()
                and np.isfinite(result.jac).all()
            )
            if result.success and finite:
                results.append(result)
            records.append(
                dict(
                    objective=float(result.fun) if np.isfinite(result.fun) else None,
                    success=bool(result.success),
                    finite=finite,
                    iterations=int(result.nit),
                    message=str(result.message),
                )
            )
            print(f"MAP {k}: {result.fun:.5f}, success={result.success}", flush=True)
        if not results:
            raise RuntimeError(f"No finite successful active-star MAP fit: {records}")
        return min(results, key=lambda r: r.fun), records

    def conditional(self, q, key):
        """Exact conditional simulation by Matheron's rule, in O(N).

        tinygp 0.3.1 cannot condition some sums of SHO kernels through its
        block-matrix covariance path. Prior simulation plus two triangular
        solves gives the same conditional Gaussian without that path.
        """
        p = self.physical(q)
        amp = self.magnification(p, self.times)
        fs = p["baseline"] * p["source_fraction"]
        lens = fs * amp + p["baseline"] - fs
        gp = build_gp(self.times, self.error / amp, p)

        def project(value):
            solved = gp.solver.solve_triangular(value)
            solved = gp.solver.solve_triangular(solved, transpose=True)
            return gp.kernel.matmul(self.times, solved)

        latent_key, noise_key = jax.random.split(key)
        prior = build_gp(self.times, jnp.zeros_like(self.times), p).sample(latent_key)
        noise = self.error / amp * jax.random.normal(noise_key, self.times.shape)
        residual = (self.flux - lens) / amp
        draw = prior + project(residual - prior - noise)
        return lens, lens + amp * draw, lens + amp * project(residual)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(problem, map_q, args, output):
    # Source-crossing time and impact ratio straighten finite-source correlations.
    # This is an exact change of coordinates, with the original q priors.
    # BEGIN ACTIVE OBJECTIVE
    def objective(y):
        q = to_prior(problem, y)
        return -problem.loglike(q) + 0.5 * jnp.sum(q * q) - log_prior_jacobian(problem, y)

    # END ACTIVE OBJECTIVE

    sampling_center = to_sampling(problem, jnp.asarray(map_q))
    hessian = np.asarray(jax.jit(jax.hessian(objective))(sampling_center))
    eig, vectors = np.linalg.eigh((hessian + hessian.T) / 2)
    transform = vectors @ np.diag(1 / np.sqrt(np.maximum(eig, 1e-4)))
    center, transform_jax = jnp.asarray(sampling_center), jnp.asarray(transform)

    # BEGIN ACTIVE NUMPYRO
    def posterior():
        z = numpyro.sample("z", dist.Normal(jnp.zeros(len(NAMES)), 1).to_event(1))
        y = center + transform_jax @ z
        numpyro.factor(
            "observations_and_prior",
            -objective(y) + 0.5 * jnp.sum(z * z),
        )

    mcmc = MCMC(
        NUTS(posterior, dense_mass=True, target_accept_prob=args.target_accept, max_tree_depth=10),
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=2,
        chain_method="sequential",
        progress_bar=args.progress,
    )
    # END ACTIVE NUMPYRO
    initial = np.random.default_rng(args.seed + 1).normal(0, 0.2, (2, len(NAMES)))
    print(f"Sampling 2 chains: {args.warmup} warmup + {args.samples} retained", flush=True)
    mcmc.run(
        jax.random.key(args.seed + 1),
        init_params={"z": jnp.asarray(initial)},
        extra_fields=("diverging", "num_steps"),
    )
    z = np.asarray(mcmc.get_samples(group_by_chain=True)["z"])
    q = np.asarray(to_prior(problem, sampling_center + z @ transform.T))
    physical = {k: np.asarray(v) for k, v in problem.physical(q).items()}
    diagnostics = chain_summary({"q": q, **physical}, group_by_chain=True)
    extras = mcmc.get_extra_fields(group_by_chain=True)
    report = dict(
        chains=2,
        warmup=args.warmup,
        samples=args.samples,
        target_accept=args.target_accept,
        max_tree_depth=10,
        coordinates=SAMPLING_COORDINATES,
        coordinate_center=np.asarray(sampling_center).tolist(),
        prior_preservation="Exact Jacobian to the original standardized Normal q coordinates; fixed linear whitening has a constant Jacobian.",
        mean_num_steps=float(np.mean(extras["num_steps"])),
        max_r_hat=float(max(np.max(v["r_hat"]) for v in diagnostics.values())),
        min_n_eff=float(min(np.min(v["n_eff"]) for v in diagnostics.values())),
        divergences=int(np.sum(extras["diverging"])),
        tree_depth_saturated=int(np.sum(np.asarray(extras["num_steps"]) >= 1023)),
    )
    report["converged"] = bool(
        report["max_r_hat"] < 1.01
        and report["min_n_eff"] >= 400
        and report["divergences"] == 0
        and report["tree_depth_saturated"] == 0
    )
    np.savez_compressed(output / "posterior.npz", q=q, **physical)
    report["posterior_sha256"] = _sha(output / "posterior.npz")
    statistics = {
        key: dict(zip(("q16", "median", "q84"), np.quantile(value, [0.16, 0.5, 0.84]).tolist()))
        for key, value in physical.items()
    }
    return q, dict(sampling=report, posterior=statistics, hessian_eigenvalues=eig.tolist())


def save_predictive(problem, result, output):
    """Pair joint parameter samples with full conditional latent GP paths."""
    sampling = result.get("sampling", {})
    if not sampling.get("converged", False):
        raise ValueError("A converged posterior is required for prediction")
    if _sha(output / "posterior.npz") != sampling.get("posterior_sha256"):
        raise ValueError("Posterior does not match the sampling record")
    if (
        _sha(output / "photometry.csv")
        != result["provenance"]["synthetic_sha256"]["photometry.csv"]
    ):
        raise ValueError("Synthetic observations do not match the sampling record")
    observed = np.genfromtxt(output / "photometry.csv", delimiter=",", names=True)
    for name, values in (
        ("time_day", problem.times),
        ("flux", problem.flux),
        ("sigma_flux", problem.error),
    ):
        if not np.allclose(observed[name], np.asarray(values), rtol=1e-12, atol=1e-13):
            raise ValueError("Current injection differs from the sampled observations")
    with np.load(output / "posterior.npz") as posterior:
        flat = posterior["q"].reshape(-1, len(NAMES))
    indices = np.linspace(0, len(flat) - 1, 128, dtype=int)
    keys = jax.random.split(jax.random.key(problem.seed + 2), len(indices))
    draw = jax.jit(lambda qq, kk: jax.lax.map(lambda x: problem.conditional(*x), (qq, kk)))
    lens, total, _ = draw(jnp.asarray(flat[indices]), keys)
    _, _, map_total = jax.jit(problem.conditional)(jnp.asarray(result["map_q"]), keys[0])
    np.savez_compressed(
        output / "predictive.npz",
        times=problem.times,
        draw_lens_flux=np.asarray(lens),
        draw_total_flux=np.asarray(total),
        truth_lens_flux=problem.lens_flux,
        truth_gp_flux=problem.gp_signal,
        observed_flux=problem.flux,
        sigma_flux=problem.error,
        map_total_flux=map_total,
    )
    result["provenance"]["predictive_sha256"] = _sha(output / "predictive.npz")


def finalize_metadata(problem, result, output):
    """Record numerical approximation checks and current reproducible inputs."""
    s = problem.settings
    amp = problem.magnification(problem.truth, problem.times)
    intrinsic = np.asarray(problem.gp_signal / amp / problem.truth["source_fraction"])
    ratio = s["spot_to_photosphere_f146_ratio"]
    coverage = s["mean_projected_spot_coverage"] - intrinsic * (
        1 - s["mean_projected_spot_coverage"] * (1 - ratio)
    ) / (1 - ratio)
    from example.inference.code.active_settings import saturation_check

    result["physical_checks"] = dict(
        saturation=saturation_check(problem.lens_flux + problem.gp_signal, s),
        realized_fractional_gp_rms=float(np.std(intrinsic)),
        equivalent_projected_spot_coverage_range=[float(coverage.min()), float(coverage.max())],
        note="Equivalent projected coverage only; the injection does not resolve a surface spot map.",
    )
    times = np.asarray(problem.times)
    exposure_times = (times[:, None] + np.asarray(problem.exposure_nodes)).ravel()
    event = sl.Parameters(t0=problem.truth["t0"], u0=problem.truth["u0"], t_e=problem.truth["t_e"])
    point = sl.Model(source="point").magnification(exposure_times, event)
    point = point.reshape(len(times), -1) @ problem.exposure_weights
    difference = problem.truth["source_fraction"] * (amp - point)
    result["physical_checks"]["finite_source"] = dict(
        rho=problem.rho,
        minimum_u_over_rho=problem.truth["u0"] / problem.rho,
        max_normalized_flux_difference=float(jnp.max(jnp.abs(difference))),
        fixed_parameter_delta_chi2=float(jnp.sum((difference / problem.error) ** 2)),
        note="Source radius rho is jointly inferred. The mean linear limb coefficient is fixed.",
    )
    from example.inference.code.active_checks import finite_source_check

    comparison, predictions = finite_source_check(problem, result["map_q"])
    result["physical_checks"]["finite_source"]["comparison"] = comparison
    np.savez_compressed(output / "point_source_comparison.npz", **predictions)
    result["provenance"]["point_source_comparison_sha256"] = _sha(
        output / "point_source_comparison.npz"
    )
    result["physical_checks"]["annual_parallax"] = physical_parallax_check(problem)
    p = problem.physical(jnp.asarray(result["map_q"]))
    amp = problem.magnification(p, problem.times)
    fs = p["baseline"] * p["source_fraction"]
    residual = (problem.flux - p["baseline"] + fs) / amp - fs
    whitened = build_gp(problem.times, problem.error / amp, p).solver.solve_triangular(residual)
    result["residual_diagnostics"] = dict(
        n=len(times),
        gp_whitened_chi2=float(jnp.sum(whitened**2)),
        map_log_likelihood=float(problem.loglike(jnp.asarray(result["map_q"]))),
        note="Cholesky innovations use the full magnified GP + measurement covariance.",
    )
    result["configuration"] = {**s, "seed": problem.seed}
    result["provenance"]["prediction_script_sha256"] = _sha(__file__)
    result["provenance"]["physical_check_script_sha256"] = _sha(
        Path(__file__).with_name("active_checks.py")
    )
    result["provenance"]["software_versions"] = {
        name: importlib.metadata.version(name)
        for name in ("jax", "jaxlib", "numpy", "numpyro", "tinygp", "scipy")
    }


def run(args):
    started = monotonic()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    problem = ActiveFFPFit(args.seed)
    if args.postprocess:
        result = json.loads((output / "summary.json").read_text())
        if result["configuration"]["seed"] != args.seed:
            raise ValueError("Seed differs from the saved inference run")
        save_predictive(problem, result, output)
        finalize_metadata(problem, result, output)
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print("Rebuilt conditional predictions and verified saved observations", flush=True)
        return
    np.savetxt(
        output / "photometry.csv",
        np.column_stack(
            [problem.times, problem.flux, problem.error, problem.lens_flux, problem.gp_signal]
        ),
        delimiter=",",
        header="time_day,flux,sigma_flux,truth_lens_flux,truth_gp_flux",
        comments="",
        fmt="%.14g",
    )
    best, records = problem.fit_map(args.starts)
    result = dict(
        description="Synthetic active K-dwarf source / finite-source sub-Earth-mass lens",
        configuration={**problem.settings, "seed": args.seed},
        truth=problem.truth,
        fixed=dict(a1=problem.a1),
        peak_mag_ab=problem.peak_mag_ab,
        map_q=best.x.tolist(),
        map_physical={k: float(v) for k, v in problem.physical(best.x).items()},
        optimization=records,
        priors=dict(
            zip(COORDINATES, np.column_stack([problem.prior_mean, problem.prior_sd]).tolist())
        ),
        parameterization="Independent Normal priors in the listed coordinates (mean, sd); q is standardized. Sampling uses peak/width coordinates with the exact prior Jacobian and fixed Hessian whitening near the original MAP.",
        provenance=dict(
            synthetic_sha256={"photometry.csv": _sha(output / "photometry.csv")},
            script_sha256=_sha(__file__),
            sampling_coordinate_script_sha256=_sha(
                Path(__file__).with_name("active_coordinates.py")
            ),
        ),
    )
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if not args.map_only:
        q, sampled = sample(problem, best.x, args, output)
        result.update(sampled)
        (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        if not result["sampling"]["converged"]:
            raise RuntimeError(f"Posterior convergence check failed: {result['sampling']}")
        save_predictive(problem, result, output)
    finalize_metadata(problem, result, output)
    result["elapsed_seconds"] = monotonic() - started
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("peak_mag_ab", "map_physical", "elapsed_seconds")}, indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--progress", action="store_true", help="Show NUTS progress bars")
    parser.add_argument("--map-only", action="store_true")
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Rebuild conditional predictions from a verified saved posterior",
    )
    run(parser.parse_args())
