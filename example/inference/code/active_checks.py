"""Numerical approximation checks for the active K-dwarf inference example."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

import singlens as sl
from example.inference.code.active_settings import white_noise


def finite_source_check(problem, map_q):
    """Refit point-source alternatives and return ``(report, predictions)``.

    The noiseless comparison profiles five event/flux parameters against the
    injected finite-source mean curve using the supplied white errors. The
    actual-data comparison refits all eight common event, blend, and GP
    parameters with their original priors. Its likelihood difference is
    evaluated at the respective MAPs; it is not a calibrated detection
    significance, a maximum-likelihood ratio, or Bayesian evidence.

    ``report`` is JSON-safe. ``predictions`` contains cadence-matched arrays
    named ``times``, ``point_lens_flux``, ``point_mean_flux`` (including the
    conditional GP mean), and ``noiseless_best_point_flux``.
    """
    # Imported here because active_ffp imports this module for diagnostics.
    from example.inference.code.active_ffp import (
        COORDINATES,
        NAMES,
        build_gp,
        gp_log_probability,
    )

    times = jnp.asarray(problem.times)
    error = np.asarray(problem.error)
    truth = problem.truth
    map_q = np.asarray(map_q, dtype=float)
    if map_q.shape != (len(COORDINATES),) or not np.isfinite(map_q).all():
        raise ValueError("Expected a finite-source MAP in the original prior coordinates")
    rho_index = COORDINATES.index("log_rho")
    common_indices = np.delete(np.arange(len(COORDINATES)), rho_index)
    exposed = times[:, None] + problem.exposure_nodes
    point_model = sl.Model(source="point")

    def point_amplification(params):
        event = sl.Parameters(t0=params["t0"], u0=params["u0"], t_e=params["t_e"])
        amp = point_model.magnification(exposed.ravel(), event)
        return amp.reshape(exposed.shape) @ problem.exposure_weights

    @jax.jit
    def point_flux(x):
        params = dict(t0=x[0], u0=jnp.exp(x[1]), t_e=jnp.exp(x[2]))
        return x[3] * point_amplification(params) + x[4]

    jacobian = jax.jit(jax.jacfwd(point_flux))
    fs = truth["baseline"] * truth["source_fraction"]
    initial = np.array(
        [truth["t0"], np.log(truth["u0"]), np.log(truth["t_e"]), fs, truth["baseline"] - fs]
    )
    target = np.asarray(problem.lens_flux)
    # Allow negative blend flux for this conservative, prior-free shape test.
    bounds = (
        [float(times.min()), np.log(1e-5), np.log(1e-5), 0.0, -np.inf],
        [float(times.max()), np.log(100), np.log(1000), np.inf, np.inf],
    )
    noiseless_candidates, noiseless_records = [], []
    for factor in (1.0, 0.5, 1.5):
        start = initial.copy()
        start[1] += np.log(factor)
        result = least_squares(
            lambda x: (np.asarray(point_flux(x)) - target) / error,
            start,
            jac=lambda x: np.asarray(jacobian(x)) / error[:, None],
            bounds=bounds,
            xtol=1e-11,
            ftol=1e-11,
            gtol=1e-8,
            max_nfev=3000,
        )
        noiseless_records.append(
            dict(
                success=bool(result.success),
                chi2=float(np.sum(result.fun**2)),
                evaluations=int(result.nfev),
                message=str(result.message),
            )
        )
        if result.success and np.isfinite(result.fun).all():
            noiseless_candidates.append(result)
    if not noiseless_candidates:
        raise RuntimeError(f"Noiseless point-source refits failed: {noiseless_records}")
    shape_fit = min(noiseless_candidates, key=lambda r: np.sum(r.fun**2))
    noiseless_point = np.asarray(point_flux(shape_fit.x))

    def physical(q_common):
        full = jnp.asarray(map_q).at[common_indices].set(q_common)
        params = problem.physical(full)
        return {name: value for name, value in params.items() if name != "rho"}

    @jax.jit
    def loglike(q_common):
        params = physical(q_common)
        return gp_log_probability(
            point_amplification(params), params, times, problem.flux, problem.error
        )

    objective = jax.jit(jax.value_and_grad(lambda q: -loglike(q) + 0.5 * jnp.sum(q * q)))
    starts = [map_q[common_indices].copy()]
    profiled = {name: float(value) for name, value in problem.physical(map_q).items()}
    x = shape_fit.x
    profiled.update(
        t0=float(x[0]),
        u0=float(np.exp(x[1])),
        t_e=float(np.exp(x[2])),
        baseline=float(max(x[3] + x[4], 1e-6)),
        source_fraction=float(np.clip(x[3] / max(x[3] + x[4], 1e-6), 0.01, 0.99)),
    )
    starts.append(np.clip(problem.pack(profiled)[common_indices], -7.9, 7.9))
    map_candidates, map_records = [], []
    for start in starts:
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            bounds=[(-8, 8)] * len(common_indices),
            options=dict(maxiter=2000, ftol=1e-12, gtol=1e-5, maxls=50),
        )
        map_records.append(
            dict(
                success=bool(result.success),
                negative_log_posterior_up_to_constant=float(result.fun),
                log_likelihood=float(loglike(result.x)),
                iterations=int(result.nit),
                max_abs_gradient=float(np.max(np.abs(result.jac))),
                message=str(result.message),
            )
        )
        if result.success and np.isfinite(result.fun):
            map_candidates.append(result)
    if not map_candidates:
        raise RuntimeError(f"Actual-data point-source MAP refits failed: {map_records}")
    point_map = min(map_candidates, key=lambda r: r.fun)
    params = physical(point_map.x)
    amp = point_amplification(params)
    source_flux = params["baseline"] * params["source_fraction"]
    lens = source_flux * amp + params["baseline"] - source_flux
    gp = build_gp(times, problem.error / amp, params)
    residual = (problem.flux - lens) / amp
    solved = gp.solver.solve_triangular(residual)
    solved = gp.solver.solve_triangular(solved, transpose=True)
    conditional_mean = lens + amp * gp.kernel.matmul(times, solved)
    point_loglike = float(loglike(point_map.x))
    finite_loglike = float(problem.loglike(map_q))
    point_parameters = {name: float(params[name]) for name in NAMES if name != "rho"}
    report = dict(
        noiseless_refitted_pspl=dict(
            profiled_delta_chi2=float(np.sum(shape_fit.fun**2)),
            fixed_parameter_delta_chi2=float(
                np.sum(((np.asarray(point_flux(initial)) - target) / error) ** 2)
            ),
            max_abs_flux_residual=float(np.max(np.abs(noiseless_point - target))),
            point_parameters=dict(
                t0=float(x[0]),
                u0=float(np.exp(x[1])),
                t_e=float(np.exp(x[2])),
                source_flux=float(x[3]),
                blend_flux=float(x[4]),
            ),
            optimization=noiseless_records,
            note="No random noise or GP in the target. Five PSPL parameters are refitted with the same supplied white errors, without priors; source flux is nonnegative and blend flux can be negative. This is a shape-mismatch diagnostic, not a calibrated significance.",
        ),
        actual_data_gp_pspl=dict(
            finite_source_log_likelihood_at_map=finite_loglike,
            point_source_log_likelihood_at_map=point_loglike,
            delta_2_log_likelihood_at_map=2 * (finite_loglike - point_loglike),
            point_map_parameters=point_parameters,
            point_map_standardized_coordinates=point_map.x.tolist(),
            coordinate_names=[COORDINATES[i] for i in common_indices],
            priors={
                COORDINATES[i]: [float(problem.prior_mean[i]), float(problem.prior_sd[i])]
                for i in common_indices
            },
            optimization=map_records,
            note="Two independent point-source MAP starts refit all eight common parameters under their original independent Normal priors in the listed transformed coordinates. Both source mean and GP covariance use exposure-averaged PSPL magnification. The reported likelihood improvement is at the separate MAPs, not at maximum likelihood; it is neither a calibrated detection significance nor Bayesian evidence.",
        ),
        n_observations=int(len(times)),
        exposure_seconds=float(problem.settings["exposure_seconds"]),
        prediction_note="point_mean_flux is the conditional latent mean at the point-source MAP, without new measurement noise or parameter uncertainty.",
    )
    predictions = dict(
        times=np.asarray(times),
        point_lens_flux=np.asarray(lens),
        point_mean_flux=np.asarray(conditional_mean),
        noiseless_best_point_flux=noiseless_point,
    )
    return report, predictions


def physical_parallax_check(problem):
    """Refit twelve annual-parallax orientations with a rectilinear light curve.

    Both curves retain the same mean finite-source profile and exposure
    integration. Known white errors alone give a conservative sensitivity
    check: the stellar GP is disabled, and no random noise is added.
    """
    settings, truth = problem.settings, problem.truth
    origin, ra, dec = 12030.0, 268.07, -29.23
    t_ref = origin + truth["t0"]
    annual = sl.Model(
        source="linear",
        solver="fft",
        parallax=sl.AnnualParallax(ra_deg=ra, dec_deg=dec, t_ref=t_ref),
    )
    times = problem.times
    exposed_times = times[:, None] + problem.exposure_nodes
    fs = truth["baseline"] * truth["source_fraction"]
    fb = truth["baseline"] - fs
    initial = np.array([truth["t0"], np.log(truth["u0"]), np.log(truth["t_e"]), fs, fb])

    @jax.jit
    def rectilinear(x):
        params = dict(t0=x[0], u0=jnp.exp(x[1]), t_e=jnp.exp(x[2]), rho=problem.rho)
        return x[3] * problem.magnification(params, times) + x[4]

    @jax.jit
    def parallax_flux(direction):
        event = sl.Parameters(
            t0=t_ref,
            u0=truth["u0"],
            t_e=truth["t_e"],
            rho=problem.rho,
            a1=problem.a1,
            pi_e_n=settings["pi_e"] * jnp.cos(direction),
            pi_e_e=settings["pi_e"] * jnp.sin(direction),
        )
        amp = annual.magnification((exposed_times + origin).ravel(), event)
        return fs * (amp.reshape(exposed_times.shape) @ problem.exposure_weights) + fb

    jacobian = jax.jit(jax.jacfwd(rectilinear))
    reference = np.asarray(rectilinear(initial))
    errors = white_noise(reference, settings)
    records = []
    for direction in np.arange(12) * (2 * np.pi / 12):
        target = np.asarray(parallax_flux(direction))
        optimum = least_squares(
            lambda x: (np.asarray(rectilinear(x)) - target) / errors,
            initial,
            jac=lambda x: np.asarray(jacobian(x)) / errors[:, None],
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-9,
        )
        if not optimum.success:
            raise RuntimeError(f"Parallax-check refit failed: {optimum.message}")
        before = reference - target
        after = np.asarray(rectilinear(optimum.x)) - target
        x = optimum.x
        records.append(
            dict(
                direction_deg=float(np.rad2deg(direction)),
                pi_e_n=float(settings["pi_e"] * np.cos(direction)),
                pi_e_e=float(settings["pi_e"] * np.sin(direction)),
                before_chi2=float(np.sum((before / errors) ** 2)),
                after_chi2=float(np.sum((after / errors) ** 2)),
                before_max_abs_flux_residual=float(np.max(np.abs(before))),
                after_max_abs_flux_residual=float(np.max(np.abs(after))),
                refit_delta=dict(
                    t0_days=float(x[0] - truth["t0"]),
                    u0=float(np.exp(x[1]) - truth["u0"]),
                    t_e_days=float(np.exp(x[2]) - truth["t_e"]),
                    source_flux=float(x[3] - fs),
                    blend_flux=float(x[4] - fb),
                    baseline=float(x[3] + x[4] - truth["baseline"]),
                    source_fraction=float(x[3] / (x[3] + x[4]) - truth["source_fraction"]),
                ),
                optimizer_success=bool(optimum.success),
            )
        )
    return dict(
        ra_deg=ra,
        dec_deg=dec,
        time_origin_hjd_tdb=2450000.0,
        relative_time_zero=origin,
        t_ref=t_ref,
        observer="Earth ephemeris; not a Roman L2 orbit",
        pi_e=float(settings["pi_e"]),
        physical_parameters={
            name: settings[name]
            for name in (
                "source_radius_solar",
                "source_distance_kpc",
                "source_theta_mas",
                "lens_mass_earth",
                "lens_distance_kpc",
                "mu_rel_masyr",
                "theta_e_mas",
            )
        },
        event={**truth, "rho": float(problem.rho), "a1": float(problem.a1)},
        n_observations=int(len(times)),
        exposure_seconds=float(settings["exposure_seconds"]),
        exposure_quadrature_nodes=int(len(problem.exposure_nodes)),
        error_model="Known white_noise(rectilinear truth); no GP covariance or random noise",
        direction_definition="pi_e_n = pi_e*cos(angle), pi_e_e = pi_e*sin(angle)",
        max_before_chi2=max(record["before_chi2"] for record in records),
        max_after_chi2=max(record["after_chi2"] for record in records),
        max_before_abs_flux_residual=max(r["before_max_abs_flux_residual"] for r in records),
        max_after_abs_flux_residual=max(r["after_max_abs_flux_residual"] for r in records),
        orientations=records,
        note=(
            "Twelve sampled directions for this sight line and epoch, not a universal bound. "
            "Five rectilinear parameters are refitted without priors: t0, log(u0), log(t_e), "
            "source flux, and blend flux. Flux residuals use total-baseline units. "
            "This checks an approximation; it does not measure planet mass or parallax."
        ),
    )
