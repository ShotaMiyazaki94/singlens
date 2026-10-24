"""Controlled recovery of a luminous G-dwarf lens from synthetic joint data.

Run from the repository root with PYTHONPATH=src:.  Earth orbital parallax,
idealized seasons and fixed independent Gaussian errors are experiment settings,
not a Roman performance forecast. No relation between fitted flux and mass is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.linalg import cho_solve
from numpyro.diagnostics import summary as chain_summary
from numpyro.infer import MCMC, NUTS
from scipy.optimize import minimize

import singlens as sl
from singlens.trajectory.ephemeris import (
    EarthOrbitalParallaxProjector,
    interp_uniform_linear,
    light_time_corrected_time,
    load_builtin_earth_ephemeris,
)

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
T_REF = 12030.0
RA_DEG, DEC_DEG = 268.07, -29.23
THETA_STAR_MAS = (695700000.0 / 149597870700.0) / 8.0
BETA_NAMES = ["east0_mas", "north0_mas", "mu_east_masyr", "mu_north_masyr", "pi_baseline_mas"]
PARAMETER_NAMES = [
    "t0",
    "log_abs_u0",
    "log_t_e",
    "pi_e_n",
    "pi_e_e",
    "log_theta_e",
    "log_source_flux",
    "log_lens_flux",
]


def marginalized_gaussian(residual, sigma, design, prior_sd):
    """Normalized likelihood after integrating a zero-mean proper Gaussian prior."""
    scaled = design * prior_sd[None, :] / sigma[:, None]
    r = residual / sigma
    precision = jnp.eye(design.shape[1]) + scaled.T @ scaled
    chol = jnp.linalg.cholesky(precision)
    mean_white = cho_solve((chol, True), scaled.T @ r)
    covariance_white = cho_solve((chol, True), jnp.eye(design.shape[1]))
    remainder = r - scaled @ mean_white
    quadratic = remainder @ remainder + mean_white @ mean_white
    logdet = 2 * jnp.log(jnp.diag(chol)).sum()
    logp = -0.5 * (quadratic + logdet + jnp.log(2 * jnp.pi * sigma**2).sum())
    return (logp, prior_sd * mean_white, prior_sd[:, None] * covariance_white * prior_sd[None, :])


class RomanJointFit:
    """Uniform-source joint likelihood with five marginalized baseline coefficients."""

    def __init__(self, *, seed=20261023, u0_sign=1, phot_times=None, ast_times=None):
        if u0_sign not in (-1, 1):
            raise ValueError("u0_sign must be -1 or +1")
        self.seed, self.sign = seed, u0_sign
        self.settings = dict(seed=seed, u0_sign=u0_sign)
        self.season_centers = T_REF + 365.25 * np.array([0, 0.5, 1, 3.5, 4, 4.5])
        self.season_edges = np.column_stack([self.season_centers - 36, self.season_centers + 36])
        default_phot = np.concatenate(
            [
                np.arange(start, stop, 1 / 120 if k == 0 else 1.0)
                for k, (start, stop) in enumerate(self.season_edges)
            ]
        )
        default_ast = np.concatenate(
            [np.arange(start + 0.5, stop, 1.0) for start, stop in self.season_edges]
        )
        self.phot_times = np.asarray(default_phot if phot_times is None else phot_times)
        self.ast_times = np.asarray(default_ast if ast_times is None else ast_times)
        is_first = (self.phot_times >= self.season_edges[0, 0]) & (
            self.phot_times < self.season_edges[0, 1]
        )
        self.phot_sigma = np.where(is_first, 0.005, 0.005 / np.sqrt(120.0))
        self.ast_sigma = np.full((len(self.ast_times), 2), 0.10)
        self.beta_truth = np.array([0.8, -0.4, -3.0, -5.0, 0.15166666666666667])
        self.beta_prior_sd = jnp.array([10.0, 10.0, 10.0, 10.0, 1.0])
        self.prior_mean = np.array(
            [T_REF, np.log(0.3), np.log(40), 0, 0, np.log(0.6), np.log(0.5), np.log(0.5)]
        )
        self.prior_sd = np.array([10.0, 0.8, 0.7, 0.2, 0.2, 0.7, 1.0, 1.0])
        theta = np.sqrt(8.144 * (1 / 6 - 1 / 8))
        pi_e = (1 / 6 - 1 / 8) / theta
        self.truth = dict(
            t0=T_REF,
            u0=0.3,
            t_e_day=365.25 * theta / 5,
            pi_e_n=pi_e * np.cos(np.pi / 6),
            pi_e_e=pi_e * np.sin(np.pi / 6),
            theta_e_mas=theta,
            source_flux=0.36,
            lens_flux=0.64,
            mass_msun=1.0,
            lens_distance_kpc=6.0,
            source_distance_kpc=8.0,
            theta_star_mas=THETA_STAR_MAS,
            rho=THETA_STAR_MAS / theta,
            pi_e=pi_e,
            pi_rel_mas=1 / 6 - 1 / 8,
            mu_geo_masyr=5.0,
        )
        truth_p = np.array(
            [
                T_REF,
                np.log(0.3),
                np.log(self.truth["t_e_day"]),
                self.truth["pi_e_n"],
                self.truth["pi_e_e"],
                np.log(theta),
                np.log(0.36),
                np.log(0.64),
            ]
        )
        self.q_truth = (truth_p - self.prior_mean) / self.prior_sd
        self.model = sl.Model(
            source="uniform",
            solver="fft",
            parallax=sl.AnnualParallax(ra_deg=RA_DEG, dec_deg=DEC_DEG, t_ref=T_REF),
        )
        eph = load_builtin_earth_ephemeris()
        self.projector = EarthOrbitalParallaxProjector(eph, RA_DEG, DEC_DEG, T_REF + 2450000.0)
        self.ast_design = self.design(self.ast_times)
        # Generate the same positive-u0 truth for either fitted sign.
        rng = np.random.default_rng(seed)
        self.sign = 1
        flux_truth, _ = self.observables(self.q_truth, self.phot_times)
        _, shift_truth = self.observables(self.q_truth, self.ast_times)
        centroid_truth = self.baseline(self.ast_times, self.beta_truth) + shift_truth
        self.sign = u0_sign
        self.phot_truth = np.asarray(flux_truth)
        self.ast_truth = np.asarray(centroid_truth)
        self.phot_flux = self.phot_truth + rng.normal(size=len(self.phot_times)) * self.phot_sigma
        self.ast_xy = self.ast_truth + rng.normal(size=self.ast_truth.shape) * self.ast_sigma
        self.loglike_jit = jax.jit(self.loglike)
        self.objective = jax.jit(
            jax.value_and_grad(lambda q: -self.loglike(q) + 0.5 * jnp.sum(q * q))
        )

    def earth_factors(self, times):
        """Full E,N parallax factors at reception JD, inverse-corrected from HJD."""
        p = self.projector
        jd = light_time_corrected_time(jnp.asarray(times) + 2450000.0, p.t0, p.dt, p.rv, p.n_hat)
        r = interp_uniform_linear(jd, p.t0, p.dt, p.rv)[..., :3]
        return -jnp.stack([r @ p.sky_east, r @ p.sky_north], axis=-1)

    def design(self, times):
        times = jnp.asarray(times)
        n = len(times)
        design = jnp.zeros((n, 2, 5))
        design = design.at[:, 0, 0].set(1.0).at[:, 1, 1].set(1.0)
        years = (times - T_REF) / 365.25
        design = design.at[:, 0, 2].set(years).at[:, 1, 3].set(years)
        return design.at[:, :, 4].set(self.earth_factors(times)).reshape(-1, 5)

    def baseline(self, times, beta):
        return (self.design(times) @ jnp.asarray(beta)).reshape(-1, 2)

    def unpack(self, q):
        p = jnp.asarray(self.prior_mean) + jnp.asarray(q) * jnp.asarray(self.prior_sd)
        theta = jnp.exp(p[5])
        event = sl.Parameters(
            t0=p[0],
            u0=self.sign * jnp.exp(p[1]),
            t_e=jnp.exp(p[2]),
            pi_e_n=p[3],
            pi_e_e=p[4],
            rho=THETA_STAR_MAS / theta,
        )
        return p, event

    def observables(self, q, times):
        """Total flux and luminous-lens shift relative to unlensed combined light."""
        # BEGIN JOINT OBSERVABLES
        p, event = self.unpack(q)
        times = jnp.asarray(times)
        w = self.model.trajectory(times, event)
        amp = self.model.magnification_at(w, rho=event.rho)
        source_shift = self.model.astrometric_shift_at(w, rho=event.rho)
        rotation = -(p[3] + 1j * p[4]) / jnp.hypot(p[3], p[4])
        theta, fs, fl = jnp.exp(p[5]), jnp.exp(p[6]), jnp.exp(p[7])
        flux = fs * amp + fl
        shift = (
            theta
            * rotation
            * (fs * amp / flux * source_shift + fs * fl * (amp - 1) / ((fs + fl) * flux) * w)
        )
        return flux, jnp.stack([shift.imag, shift.real], axis=-1)
        # END JOINT OBSERVABLES

    def conditional_beta(self, q):
        _, shift = self.observables(q, self.ast_times)
        _, mean, covariance = marginalized_gaussian(
            (jnp.asarray(self.ast_xy) - shift).ravel(),
            jnp.asarray(self.ast_sigma).ravel(),
            self.ast_design,
            self.beta_prior_sd,
        )
        return mean, covariance

    def predict(self, q, times, beta=None):
        flux, shift = self.observables(q, times)
        if beta is None:
            beta, _ = self.conditional_beta(q)
        return dict(
            flux=flux, lensing_shift_mas=shift, centroid_mas=self.baseline(times, beta) + shift
        )

    # BEGIN JOINT LIKELIHOOD
    def loglike(self, q):
        flux, _ = self.observables(q, self.phot_times)
        phot = -0.5 * jnp.sum(
            ((jnp.asarray(self.phot_flux) - flux) / self.phot_sigma) ** 2
            + jnp.log(2 * jnp.pi * self.phot_sigma**2)
        )
        _, shift = self.observables(q, self.ast_times)
        ast, _, _ = marginalized_gaussian(
            (jnp.asarray(self.ast_xy) - shift).ravel(),
            jnp.asarray(self.ast_sigma).ravel(),
            self.ast_design,
            self.beta_prior_sd,
        )
        return phot + ast

    # END JOINT LIKELIHOOD

    def physical(self, q):
        p = self.prior_mean + np.asarray(q) * self.prior_sd
        theta, te = np.exp(p[..., 5]), np.exp(p[..., 2])
        pi_e = np.hypot(p[..., 3], p[..., 4])
        fs, fl = np.exp(p[..., 6]), np.exp(p[..., 7])
        return dict(
            t0=p[..., 0],
            u0=self.sign * np.exp(p[..., 1]),
            t_e_day=te,
            pi_e_n=p[..., 3],
            pi_e_e=p[..., 4],
            pi_e=pi_e,
            theta_e_mas=theta,
            source_flux=fs,
            lens_flux=fl,
            lens_to_source_flux=fl / fs,
            mass_msun=theta / (8.144 * pi_e),
            pi_rel_mas=theta * pi_e,
            rho=THETA_STAR_MAS / theta,
            mu_geo_masyr=365.25 * theta / te,
        )

    def residual_diagnostics(self, q):
        flux, _ = self.observables(q, self.phot_times)
        centroid = self.predict(q, self.ast_times)["centroid_mas"]
        return dict(
            photometry_chi2=float(np.sum(((self.phot_flux - flux) / self.phot_sigma) ** 2)),
            astrometry_chi2=float(np.sum(((self.ast_xy - centroid) / self.ast_sigma) ** 2)),
            photometry_n=len(self.phot_times),
            astrometry_coordinates=self.ast_xy.size,
        )

    def initial(self, k=0):
        p = self.prior_mean.copy()
        # Varied generic initial guesses; injected truth is not used for inference.
        angles = np.deg2rad([35.0, 150.0, 260.0, 330.0, 75.0, 210.0])
        p[3:5] = [0.08 * np.cos(angles[k % 6]), 0.08 * np.sin(angles[k % 6])]
        p[2] = np.log([35.0, 60.0, 25.0][k % 3])
        p[6:8] = np.log([0.4, 0.6])
        return (p - self.prior_mean) / self.prior_sd

    def fit_map(self, starts=4):
        records, results = [], []
        for k in range(starts):
            result = minimize(
                self.objective,
                self.initial(k),
                jac=True,
                method="L-BFGS-B",
                bounds=[(-10, 10)] * 8,
                options=dict(maxiter=2500, ftol=1e-11, gtol=1e-5, maxls=50),
            )
            finite = bool(
                np.isfinite(result.fun)
                and np.isfinite(result.x).all()
                and np.isfinite(result.jac).all()
            )
            records.append(
                dict(
                    objective=float(result.fun) if np.isfinite(result.fun) else None,
                    success=bool(result.success),
                    finite=finite,
                    message=str(result.message),
                    iterations=int(result.nit),
                )
            )
            if result.success and finite:
                results.append(result)
            print(
                f"u0 sign {self.sign:+} start {k}: {result.fun:.4f}, success={result.success}",
                flush=True,
            )
        if not results:
            raise RuntimeError(f"No finite successful MAP for u0 sign {self.sign:+}: {records}")
        best = min(results, key=lambda result: result.fun)
        return best, records


def _statistics(values):
    return {
        key: dict(
            q16=float(np.quantile(value, 0.16)),
            median=float(np.median(value)),
            q84=float(np.quantile(value, 0.84)),
        )
        for key, value in values.items()
    }


def save_synthetic(problem, output):
    np.savetxt(
        output / "photometry.csv",
        np.column_stack(
            [problem.phot_times, problem.phot_flux, problem.phot_sigma, problem.phot_truth]
        ),
        delimiter=",",
        header="time_hjd_tdb_minus_2450000,flux,sigma_flux,truth_flux",
        comments="",
        fmt="%.12g",
    )
    np.savetxt(
        output / "astrometry.csv",
        np.column_stack([problem.ast_times, problem.ast_xy, problem.ast_sigma, problem.ast_truth]),
        delimiter=",",
        header="time_hjd_tdb_minus_2450000,east_mas,north_mas,sigma_east_mas,sigma_north_mas,truth_east_mas,truth_north_mas",
        comments="",
        fmt="%.12g",
    )


def sample_branch(problem, map_q, args, output, label):
    def objective(q):
        return -problem.loglike(q) + 0.5 * jnp.sum(q * q)

    hessian = np.asarray(jax.jit(jax.hessian(objective))(jnp.asarray(map_q)))
    eig, vectors = np.linalg.eigh((hessian + hessian.T) / 2)
    transform = vectors @ np.diag(1 / np.sqrt(np.maximum(eig, 1e-4)))
    center, transform_jax = jnp.asarray(map_q), jnp.asarray(transform)

    # BEGIN JOINT NUMPYRO
    def posterior():
        z = numpyro.sample("z", dist.Normal(jnp.zeros(8), 1).to_event(1))
        q = center + transform_jax @ z
        numpyro.factor(
            "observations_and_prior",
            problem.loglike(q) - 0.5 * jnp.sum(q * q) + 0.5 * jnp.sum(z * z),
        )

    sampler = MCMC(
        NUTS(posterior, dense_mass=True, target_accept_prob=args.target_accept, max_tree_depth=10),
        num_warmup=args.warmup,
        num_samples=args.samples,
        num_chains=args.chains,
        chain_method="sequential",
        progress_bar=False,
    )
    # END JOINT NUMPYRO
    initial = np.random.default_rng(args.seed + problem.sign).normal(0, 0.2, (args.chains, 8))
    print(f"Sampling {label}: {args.chains} chains, {args.warmup}+{args.samples}", flush=True)
    sampler.run(
        jax.random.PRNGKey(args.seed + int(problem.sign < 0)),
        init_params={"z": jnp.asarray(initial if args.chains > 1 else initial[0])},
        extra_fields=("diverging", "num_steps"),
    )
    z = np.asarray(sampler.get_samples(group_by_chain=True)["z"])
    q = map_q + z @ transform.T
    physical = problem.physical(q)
    beta_mean, beta_cov = jax.jit(jax.vmap(problem.conditional_beta))(jnp.asarray(q.reshape(-1, 8)))
    covariance = np.asarray(beta_cov)
    rng = np.random.default_rng(args.seed + 180 + problem.sign)
    beta = np.asarray(beta_mean) + np.einsum(
        "nij,nj->ni", np.linalg.cholesky(covariance), rng.normal(size=beta_mean.shape)
    )
    beta = beta.reshape(args.chains, args.samples, 5)
    diagnostics = chain_summary({"q": q, **physical}, group_by_chain=True)
    extras = sampler.get_extra_fields(group_by_chain=True)
    sampling = dict(
        chains=args.chains,
        warmup=args.warmup,
        samples=args.samples,
        seed=args.seed,
        target_accept=args.target_accept,
        max_tree_depth=10,
        max_r_hat=float(max(np.max(value["r_hat"]) for value in diagnostics.values())),
        min_n_eff=float(min(np.min(value["n_eff"]) for value in diagnostics.values())),
        divergences=int(np.sum(extras["diverging"])),
        tree_depth_saturated=int(np.sum(np.asarray(extras["num_steps"]) >= 1023)),
    )
    sampling["converged"] = bool(
        sampling["max_r_hat"] < 1.01
        and sampling["min_n_eff"] >= 400
        and sampling["divergences"] == 0
        and sampling["tree_depth_saturated"] == 0
    )
    np.savez_compressed(
        output / f"posterior_{label}.npz", q=q, beta=beta, branch=np.array(label), **physical
    )
    sampling["posterior_sha256"] = hashlib.sha256(
        (output / f"posterior_{label}.npz").read_bytes()
    ).hexdigest()
    return dict(
        posterior=_statistics(physical),
        baseline_posterior=_statistics(dict(zip(BETA_NAMES, np.moveaxis(beta, -1, 0)))),
        sampling=sampling,
        hessian_eigenvalues=eig.tolist(),
    )


def save_predictive(problem, branch, output, label):
    grid = np.unique(np.r_[np.linspace(T_REF - 40, T_REF + 40, 601), problem.ast_times])
    q = np.asarray(branch["map_q"])
    prediction = problem.predict(q, grid)
    sign = problem.sign
    problem.sign = 1
    truth = problem.predict(problem.q_truth, grid, problem.beta_truth)
    problem.sign = sign
    arrays = dict(
        grid_times=grid,
        selected_branch=np.array(label),
        phot_times=problem.phot_times,
        phot_flux=problem.phot_flux,
        phot_sigma=problem.phot_sigma,
        ast_times=problem.ast_times,
        ast_xy=problem.ast_xy,
        ast_sigma=problem.ast_sigma,
        season_edges=problem.season_edges,
    )
    for key in prediction:
        arrays[f"map_{key}"] = np.asarray(prediction[key])
        arrays[f"truth_{key}"] = np.asarray(truth[key])
    path = output / f"posterior_{label}.npz"
    sampling = branch.get("sampling", {})
    if sampling.get("converged", False):
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != sampling.get(
            "posterior_sha256"
        ):
            raise ValueError(f"Posterior archive does not match current {label} sampling record")
        with np.load(path) as posterior:
            q_all, beta_all = posterior["q"].reshape(-1, 8), posterior["beta"].reshape(-1, 5)
            indices = np.linspace(0, len(q_all) - 1, min(128, len(q_all)), dtype=int)
            draws = jax.jit(jax.vmap(lambda q, b: problem.predict(q, grid, b)))(
                jnp.asarray(q_all[indices]), jnp.asarray(beta_all[indices])
            )
        for key, value in draws.items():
            arrays[f"draw_{key}"] = np.asarray(value)
    np.savez_compressed(output / "predictive.npz", **arrays)


def finalize_metadata(summary, output):
    """Add reproducible source provenance and checks without changing sampled data."""
    problem = RomanJointFit(seed=summary["configuration"]["seed"])
    grid = np.linspace(
        T_REF - 4 * problem.truth["t_e_day"], T_REF + 4 * problem.truth["t_e_day"], 4801
    )
    flux_fs, shift_fs = problem.observables(problem.q_truth, grid)
    p, event = problem.unpack(problem.q_truth)
    w = problem.model.trajectory(grid, event)
    point = sl.Model(source="point")
    amp = point.magnification_at(w)
    shift = point.astrometric_shift_at(w)
    fs, fl, theta = 0.36, 0.64, problem.truth["theta_e_mas"]
    flux_ps = fs * amp + fl
    rotation = -(p[3] + 1j * p[4]) / jnp.hypot(p[3], p[4])
    displacement = (
        theta
        * rotation
        * (fs * amp / flux_ps * shift + fs * fl * (amp - 1) / ((fs + fl) * flux_ps) * w)
    )
    shift_ps = jnp.stack([displacement.imag, displacement.real], axis=-1)
    summary["finite_source_check"] = dict(
        minimum_u_over_rho=float(jnp.min(jnp.abs(w)) / event.rho),
        max_abs_flux_difference=float(jnp.max(jnp.abs(flux_fs - flux_ps))),
        max_centroid_difference_microas=float(
            1000 * jnp.max(jnp.linalg.norm(shift_fs - shift_ps, axis=-1))
        ),
        max_blended_shift_microas=float(1000 * jnp.max(jnp.linalg.norm(shift_fs, axis=-1))),
        note="Uniform finite source retained; comparison at injected parameters on ±4tE dense grid.",
    )
    for label, branch in summary["branches"].items():
        fit = problem if label == "positive" else RomanJointFit(seed=problem.seed, u0_sign=-1)
        branch["map_residuals"] = fit.residual_diagnostics(np.asarray(branch["map_q"]))
    best = summary["branches"][summary["selected_branch"]]["map_residuals"]
    best_chi2 = best["photometry_chi2"] + best["astrometry_chi2"]
    for branch in summary["branches"].values():
        residuals = branch["map_residuals"]
        branch["delta_chi2_at_map"] = (
            residuals["photometry_chi2"] + residuals["astrometry_chi2"] - best_chi2
        )
    source = Path(sl.__file__).parent
    summary["provenance"].update(
        source_sha256={
            str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(source.rglob("*.py"))
        },
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        synthetic_sha256={
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("photometry.csv", "astrometry.csv")
        },
    )
    summary["parameterization"] = (
        "Each listed prior pair is (mean, standard deviation) for an independent Normal "
        "in the named coordinate; logarithms are natural. "
        "log parameters imply lognormal physical priors and Gaussian parallax components imply "
        "a Rayleigh norm prior. q=(p-mean)/sd has independent standard Normal priors. "
        "Sampling uses q=q_MAP+L z with a fixed MAP-Hessian transform; its constant Jacobian "
        "does not change the normalized posterior. Baseline coefficients are marginalized "
        "with independent zero-mean Normal priors and the listed baseline_prior_sd, "
        "then conditionally drawn for each nonlinear sample."
    )


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    branches, problems = {}, {}
    for sign in (1, -1):
        problem = RomanJointFit(seed=args.seed, u0_sign=sign)
        label = "positive" if sign > 0 else "negative"
        problems[label] = problem
        best, records = problem.fit_map(args.starts)
        beta, _ = problem.conditional_beta(best.x)
        branches[label] = dict(
            settings=problem.settings,
            map_q=best.x.tolist(),
            map_beta=np.asarray(beta).tolist(),
            map_physical={key: float(value) for key, value in problem.physical(best.x).items()},
            optimization=records,
            map_log_likelihood=float(problem.loglike_jit(best.x)),
            map_log_posterior_unnormalized=float(-best.fun),
        )
    selected = max(branches, key=lambda label: branches[label]["map_log_posterior_unnormalized"])
    reference = branches[selected]["map_log_posterior_unnormalized"]
    problem = problems[selected]
    save_synthetic(problem, output)
    summary = dict(
        description="Synthetic G-dwarf joint photometric and full-centroid recovery",
        truth=problem.truth,
        beta_truth=dict(zip(BETA_NAMES, problem.beta_truth.tolist())),
        selected_branch=selected,
        branches=branches,
        priors=dict(
            zip(PARAMETER_NAMES, np.column_stack([problem.prior_mean, problem.prior_sd]).tolist())
        ),
        baseline_prior_sd=dict(zip(BETA_NAMES, np.asarray(problem.beta_prior_sd).tolist())),
        configuration=dict(
            seed=args.seed,
            source="uniform",
            fixed_theta_star_mas=THETA_STAR_MAS,
            time="HJD TDB - 2450000; inverse light-time correction for full Earth parallax factors",
            ra_deg=RA_DEG,
            dec_deg=DEC_DEG,
            t_ref=T_REF,
            season_edges=problem.season_edges.tolist(),
            photometry_n=len(problem.phot_times),
            astrometry_epochs=len(problem.ast_times),
            event_cadence_min=12,
            other_season_cadence_day=1,
            event_flux_sigma=0.005,
            other_flux_sigma=0.005 / np.sqrt(120),
            astrometry_sigma_mas=0.10,
            noise="Known independent Gaussian errors; daily flux samples are assigned the precision of 120 independent 12-minute measurements; no jitter/GP",
            geometry="Earth observer approximation, no Roman L2 orbit or visibility forecast",
            luminosity="Injection only sets FS=.36 FL=.64; both inferred independently with no mass-luminosity relation",
            competitive_delta_log_posterior=args.branch_threshold,
            sampled_both_signs=not args.skip_disfavored,
        ),
        software=dict(jax=jax.__version__, numpyro=numpyro.__version__, numpy=np.__version__),
        provenance=dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
    )
    for label, branch in branches.items():
        branch["delta_map_log_posterior"] = reference - branch["map_log_posterior_unnormalized"]
        branch["competitive"] = branch["delta_map_log_posterior"] <= args.branch_threshold
        if args.samples and (branch["competitive"] or not args.skip_disfavored):
            branch_args = argparse.Namespace(**vars(args))
            if label == "negative":
                branch_args.warmup = args.negative_warmup
                branch_args.target_accept = args.negative_target_accept
            branch.update(
                sample_branch(
                    problems[label], np.asarray(branch["map_q"]), branch_args, output, label
                )
            )
        summary["elapsed_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        print(
            label,
            branch.get("sampling", {}),
            branch.get("posterior", {}).get("mass_msun"),
            flush=True,
        )
        sampling = branch.get("sampling")
        if sampling is not None and not sampling["converged"]:
            raise RuntimeError(
                f"{label} posterior convergence check failed; diagnostics saved in "
                f"{output / 'summary.json'}: {sampling}"
            )
    save_predictive(problem, branches[selected], output, selected)
    finalize_metadata(summary, output)
    summary["elapsed_seconds"] = time.monotonic() - started
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "outputs"))
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--starts", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--samples", type=int, default=1200)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--negative-warmup", type=int, default=1200)
    parser.add_argument("--negative-target-accept", type=float, default=0.98)
    parser.add_argument("--branch-threshold", type=float, default=15.0)
    parser.add_argument(
        "--skip-disfavored", action="store_true", help="By default sample both signs separately"
    )
    run(parser.parse_args())
