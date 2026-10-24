"""Finite-exposure photometry of a physically specified planetary lens event."""

import hashlib
import inspect
import json
from functools import cache
from importlib import import_module
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from example._common import metadata, model_solver_metadata
from example._physical import photometric_error, planet_configuration, saturation_check
from singlens import Model, Parameters
from singlens.photometry import marginalize_flux

CONFIG = planet_configuration()
NAMES = ("t0", "u0", "t_e", "rho", "f_s", "f_b")
LENS_NAMES = NAMES[:4]
LABELS = (r"$t_0$", r"$u_0$", r"$t_E$", r"$\rho$", r"$f_s$", r"$f_b$")
TRANSFORM_NAMES = (
    "t0",
    "log_u0_over_rho",
    "log_source_crossing_time",
    "log_rho",
)
# Priors describe a detected short transit and an independently normalized baseline.
# They are densities in these coordinates, not flat physical-parameter priors.
PRIOR_LOC = np.array([0.0, np.log(0.7), np.log(0.075), np.log(0.6)])
PRIOR_SCALE = np.array([0.25, 0.6, 0.7, 0.7])
FLUX_PRIOR_MEAN = np.array([0.8, 0.2])
FLUX_PRIOR_COVARIANCE = np.array([[0.0272, -0.0252], [-0.0252, 0.0257]])
DATA_SEED = 43
EXPOSURE_ORDER = 3
INJECTION_ORDER = 9


class Injection(NamedTuple):
    times: jax.Array
    flux: jax.Array
    error: jax.Array
    truth: dict
    lens_flux: jax.Array
    gp_signal: jax.Array
    seed: int = DATA_SEED
    reference_checks: dict | None = None


def unpack(theta):
    rho, crossing_time = jnp.exp(theta[3]), jnp.exp(theta[2])
    return dict(
        t0=theta[0],
        u0=rho * jnp.exp(theta[1]),
        t_e=crossing_time / rho,
        rho=rho,
    )


def event_parameters(params):
    return Parameters(
        **{name: params[name] for name in ("t0", "u0", "t_e", "rho")},
        a1=CONFIG["event"]["a1"],
    )


def make_model(solver="table"):
    """Prepare the fixed linear profile; the short event uses a rectilinear trajectory."""
    return Model(source="linear", solver=solver)


@cache
def exposure_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes * CONFIG["exposure_seconds"] / (2 * 86400), weights / 2


def exposure_times(times, *, order=EXPOSURE_ORDER):
    offsets, _ = exposure_rule(order)
    return jnp.asarray(times)[..., None] + offsets


def magnification(model, times, params, *, order=EXPOSURE_ORDER):
    """The same fixed exposure quadrature is used by all fitted backends."""
    _, weights = exposure_rule(order)
    values = model.magnification(exposure_times(times, order=order), event_parameters(params))
    return jnp.sum(values * weights, axis=-1)


def _wing_integral(u, rho, a1, order):
    """Independent source-centered integration of the exact point-lens kernel."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    angle = (nodes + 1) * np.pi / 4
    radius, mu = np.sin(angle), np.cos(angle)
    radial_weights = weights * radius * mu * (1 - a1 + a1 * mu)
    radial_weights /= radial_weights.sum()
    cosine = np.cos(np.arange(2 * order) * np.pi / order)
    result = np.empty(len(u))
    for start in range(0, len(u), 128):
        center = u[start : start + 128, None, None]
        r = rho * radius[None, :, None]
        distance = np.sqrt(center**2 + r**2 - 2 * center * r * cosine)
        point = (distance**2 + 2) / (distance * np.hypot(distance, 2))
        result[start : start + 128] = point.mean(axis=-1) @ radial_weights
    return result


def reference_magnification(times, params, *, order=INJECTION_ORDER, spatial_order=32):
    """Independent finite-exposure injection, without tested tables or FFTLog."""
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    offsets, weights = exposure_rule(order)
    subtimes = np.asarray(times)[..., None] + offsets
    u = np.hypot((subtimes - float(params["t0"])) / float(params["t_e"]), float(params["u0"]))
    flat = u.reshape(-1)
    rho, a1 = float(params["rho"]), CONFIG["event"]["a1"]
    near = flat / rho <= 3
    values = np.empty_like(flat)
    values[near] = [integral(float(x), rho, a1=a1, tolerance=1e-11) for x in flat[near]]
    values[~near] = _wing_integral(flat[~near], rho, a1, spatial_order)
    return (values.reshape(u.shape) * weights).sum(axis=-1)


def observation_times():
    cadence = CONFIG["cadence_minutes"] / 1440
    half_exposure = CONFIG["exposure_seconds"] / (2 * 86400)
    return np.arange(
        -CONFIG["season_days"] / 2 + half_exposure,
        CONFIG["season_days"] / 2 - half_exposure,
        cadence,
    )


def reference_provenance():
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    sources = {
        function.__name__: hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        for function in (
            integral,
            _wing_integral,
            reference_magnification,
            exposure_rule,
            observation_times,
            photometric_error,
            saturation_check,
        )
    }
    configuration = json.dumps(CONFIG, sort_keys=True, separators=(",", ":"))
    return dict(
        source_sha256=sources,
        configuration_sha256=hashlib.sha256(configuration.encode()).hexdigest(),
        fit_exposure_order=EXPOSURE_ORDER,
        injection_exposure_order=INJECTION_ORDER,
    )


def reference_data_hash(times, error, lens_flux):
    values = np.column_stack([times, error, lens_flux]).astype("<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _validate_reference_data(times, error, mean, checks):
    arrays = [np.asarray(value) for value in (times, error, mean)]
    if any(value.ndim != 1 or value.size == 0 for value in arrays):
        raise ValueError("Independent reference arrays must be nonempty and one-dimensional")
    if len({value.shape for value in arrays}) != 1:
        raise ValueError("Independent reference array shapes differ")
    if any(not np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("Independent reference arrays must be finite")
    if np.any(arrays[1] <= 0) or np.any(arrays[2] <= 0):
        raise ValueError("Independent reference errors and mean flux must be positive")
    if not np.array_equal(arrays[0], observation_times()):
        raise ValueError("Independent reference cache has different observation times")
    if checks.get("saturation", {}).get("passed") is not True:
        raise ValueError("Independent reference cache failed its brightness checks")
    for name, maximum in (
        ("injection_refinement_max_flux_error_sigma", 1e-3),
        ("fitting_exposure_rule_max_flux_error_sigma", 0.01),
        ("overlap_max_relative_difference", 1e-9),
    ):
        value = checks.get(name, np.nan)
        if not np.isfinite(value) or value < 0 or value > maximum:
            raise ValueError(f"Independent reference cache failed validation: {name}")


def save_reference_data(path, *, times, error, lens_flux, checks):
    """Save an independently validated mean once, before timing any fit."""
    _validate_reference_data(times, error, lens_flux, checks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    description = dict(
        provenance=reference_provenance(),
        checks=checks,
        numeric_payload_sha256=reference_data_hash(times, error, lens_flux),
    )
    np.savez_compressed(
        path,
        times=np.asarray(times),
        error=np.asarray(error),
        lens_flux=np.asarray(lens_flux),
        metadata=json.dumps(description, sort_keys=True),
    )


def load_reference_data(path):
    with np.load(path, allow_pickle=False) as saved:
        times, error, mean = (saved[name].copy() for name in ("times", "error", "lens_flux"))
        description = json.loads(str(saved["metadata"]))
    if description["provenance"] != reference_provenance():
        raise ValueError("Independent reference cache has different configuration or source hashes")
    _validate_reference_data(times, error, mean, description["checks"])
    if description["numeric_payload_sha256"] != reference_data_hash(times, error, mean):
        raise ValueError("Independent reference cache checksum mismatch")
    return times, error, mean, description["checks"]


def _noisy_data(times, error, lens_flux, truth, checks, seed):
    flux = lens_flux + np.random.default_rng(seed).normal(size=len(times)) * error
    return Injection(
        jnp.asarray(times),
        jnp.asarray(flux),
        jnp.asarray(error),
        truth,
        jnp.asarray(lens_flux),
        jnp.zeros(len(times)),
        seed,
        checks,
    )


def synthetic_data(*, seed=DATA_SEED, reference=None):
    """One independent-integration mean, with a reproducible Gaussian noise draw."""
    truth = {name: float(CONFIG["event"][name]) for name in NAMES}
    if reference is not None and Path(reference).exists():
        times, error, lens_flux, checks = load_reference_data(reference)
        return _noisy_data(times, error, lens_flux, truth, checks, seed)
    times = observation_times()
    mean_a = reference_magnification(times, truth)
    lens_flux = truth["f_s"] * mean_a + truth["f_b"]
    error = np.asarray(photometric_error(lens_flux, CONFIG))
    near = np.flatnonzero(np.abs(times - truth["t0"]) < 4 * truth["t_e"])
    selected = np.unique(np.r_[near, np.linspace(0, len(times) - 1, 24, dtype=int)])
    refined = reference_magnification(times[selected], truth, order=15, spatial_order=64)
    coarse = reference_magnification(times[selected], truth, order=EXPOSURE_ORDER)
    scale = truth["f_s"] / error[selected]
    overlap_u = truth["rho"] * np.array([2.8, 3.0, 3.2])
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    overlap_reference = np.array(
        [integral(x, truth["rho"], a1=CONFIG["event"]["a1"], tolerance=1e-11) for x in overlap_u]
    )
    overlap = _wing_integral(overlap_u, truth["rho"], CONFIG["event"]["a1"], 32)
    checks = {
        "saturation": saturation_check(lens_flux, CONFIG),
        "selected_epochs": selected.tolist(),
        "injection_refinement_max_flux_error_sigma": float(
            np.max(np.abs((mean_a[selected] - refined) * scale))
        ),
        "fitting_exposure_rule_max_flux_error_sigma": float(
            np.max(np.abs((coarse - refined) * scale))
        ),
        "overlap_max_relative_difference": float(np.max(np.abs(overlap / overlap_reference - 1))),
        "injection_order": INJECTION_ORDER,
        "refined_exposure_order": 15,
        "wing_orders": [32, 64],
        "wing_boundary_u_over_rho": 3,
    }
    if not checks["saturation"]["passed"]:
        raise RuntimeError(
            f"Injection violates its approximate brightness guards: {checks['saturation']}"
        )
    if checks["injection_refinement_max_flux_error_sigma"] > 1e-3:
        raise RuntimeError(f"Independent injection needs more quadrature: {checks}")
    if checks["fitting_exposure_rule_max_flux_error_sigma"] > 0.01:
        raise RuntimeError(f"Fixed exposure rule needs refinement: {checks}")
    if checks["overlap_max_relative_difference"] > 1e-9:
        raise RuntimeError(f"Independent spatial references disagree: {checks}")
    if reference is not None:
        save_reference_data(reference, times=times, error=error, lens_flux=lens_flux, checks=checks)
    return _noisy_data(times, error, lens_flux, truth, checks, seed)


def synthetic_event(*, solver="table", seed=DATA_SEED, reference=None):
    return make_model(solver), synthetic_data(seed=seed, reference=reference)


def data_hash(data):
    values = np.column_stack([data.times, data.flux, data.error, data.lens_flux]).astype("<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def injection_discrepancy(model, data):
    params = data.truth
    predicted = params["f_s"] * magnification(model, data.times, params) + params["f_b"]
    difference = np.asarray((predicted - data.lens_flux) / data.error)
    fit_residual = np.asarray((data.flux - predicted) / data.error)
    injection_residual = np.asarray((data.flux - data.lens_flux) / data.error)
    return dict(
        scope="All epochs at injected lens and fixed injected flux parameters; finite-exposure fit minus independent injection. delta_log_likelihood is the conditional likelihood difference at those fixed fluxes, not the marginalized target.",
        max_abs_flux_difference_over_error=float(np.max(np.abs(difference))),
        rms_flux_difference_over_error=float(np.sqrt(np.mean(difference**2))),
        sum_squared_flux_difference_over_error=float(np.sum(difference**2)),
        delta_log_likelihood=float(-0.5 * np.sum(fit_residual**2 - injection_residual**2)),
    )


def flux_conditional(magnification, flux, error):
    return marginalize_flux(
        magnification,
        flux,
        error,
        prior_mean=FLUX_PRIOR_MEAN,
        prior_covariance=FLUX_PRIOR_COVARIANCE,
    )


def log_likelihood(magnification, flux, error):
    return flux_conditional(magnification, flux, error).log_likelihood


def log_prior(theta):
    # All methods share the table's physical domain; normalization is constant.
    max_ratio = jnp.hypot(
        (CONFIG["season_days"] / 2 + jnp.abs(theta[0]) + CONFIG["exposure_seconds"] / 86400)
        / jnp.exp(theta[2]),
        jnp.exp(theta[1]),
    )
    valid = (
        jnp.all(jnp.isfinite(theta))
        & (theta[3] >= np.log(1e-6))
        & (theta[3] <= np.log(10.0))
        & (max_ratio <= 1e8)
    )
    density = -0.5 * jnp.sum(((theta - PRIOR_LOC) / PRIOR_SCALE) ** 2)
    return jnp.where(valid, density, -jnp.inf)


def negative_log_posterior(theta, model, times, flux, error):
    params = unpack(theta)
    values = magnification(model, times, params)
    result = -log_likelihood(values, flux, error) - log_prior(theta)
    return jnp.where(jnp.isfinite(result), result, jnp.inf)


def fit_event(model, times, flux, error):
    """Fit the four-dimensional marginal posterior and its actual Hessian.

    A conditional-flux profile fit supplies a data-driven starting point. The
    final optimization includes the geometry-dependent marginal determinant.
    """
    prior_chol = jnp.linalg.cholesky(jnp.asarray(FLUX_PRIOR_COVARIANCE))

    def residual(theta):
        params = unpack(theta)
        values = magnification(model, times, params)
        conditional = flux_conditional(values, flux, error)
        curve = conditional.mean[0] * values + conditional.mean[1]
        flux_penalty = jax.scipy.linalg.solve_triangular(
            prior_chol, conditional.mean - FLUX_PRIOR_MEAN, lower=True
        )
        return jnp.r_[(curve - flux) / error, flux_penalty, (theta - PRIOR_LOC) / PRIOR_SCALE]

    times_np, flux_np = np.asarray(times), np.asarray(flux)
    baseline = float(np.median(flux_np))
    peak = int(np.argmax(flux_np))
    half_peak = flux_np > baseline + (flux_np[peak] - baseline) / 2
    width = (
        float(np.ptp(times_np[half_peak]))
        if half_peak.sum() > 1
        else CONFIG["cadence_minutes"] / 1440
    )
    initial = np.array(PRIOR_LOC)
    initial[0], initial[2] = (
        times_np[peak],
        np.log(max(width / 2, 0.01)),
    )
    evaluate, jacobian = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    lower, upper = np.full(4, -np.inf), np.full(4, np.inf)
    lower[3], upper[3] = np.log(1e-6), np.log(10.0)
    profile = least_squares(
        lambda x: np.asarray(evaluate(x)),
        initial,
        jac=lambda x: np.asarray(jacobian(x)),
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-8,
        max_nfev=400,
    )
    if not profile.success:
        raise RuntimeError(f"Marginal MAP initialization failed: {profile.message}")
    center = profile.x
    scale = np.linalg.cholesky(np.linalg.inv(profile.jac.T @ profile.jac))

    def objective(theta):
        return negative_log_posterior(theta, model, times, flux, error)

    whitened = jax.jit(jax.value_and_grad(lambda z: objective(center + scale @ z)))
    result = minimize(
        lambda z: jax.tree.map(np.asarray, whitened(z)),
        np.zeros(4),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 200},
    )
    # A finite objective with a small whitened gradient is a useful convergence
    # criterion when BFGS reports floating-point precision loss near the optimum.
    gradient = np.asarray(whitened(result.x)[1])
    result.success = bool(np.isfinite(result.fun) and np.max(abs(gradient)) < 1e-5)
    result.x = center + scale @ result.x
    hessian = np.asarray(jax.jit(jax.hessian(objective))(result.x))
    hessian = (hessian + hessian.T) / 2
    np.linalg.cholesky(hessian)
    result.hess_inv = np.linalg.inv(hessian)
    result.marginal_hessian = hessian
    result.profile_nfev = profile.nfev
    result.whitened_gradient_max = float(np.max(abs(gradient)))
    params = unpack(result.x)
    conditional = flux_conditional(magnification(model, times, params), flux, error)
    return result, dict(**params, f_s=conditional.mean[0], f_b=conditional.mean[1])


def experiment_metadata(model, data):
    return {
        **metadata(),
        "data_seed": data.seed,
        "data_sha256": data_hash(data),
        "reference_data_sha256": reference_data_hash(data.times, data.error, data.lens_flux),
        "reference_provenance": reference_provenance(),
        "observations": int(data.times.size),
        "physical_configuration": CONFIG,
        "cadence_minutes": CONFIG["cadence_minutes"],
        "season_days": CONFIG["season_days"],
        "exposure_seconds": CONFIG["exposure_seconds"],
        "fit_exposure_order": EXPOSURE_ORDER,
        "band": CONFIG["band"],
        "source": model.source,
        "fit_solver": model_solver_metadata(model),
        "injection_solver": "Independent lens-centered/source-centered brightness quadrature",
        "injection_scope": "Common finite-exposure mean; only the independent Gaussian noise seed varies",
        "reference_checks": data.reference_checks,
        "fixed_parameters": {"a1": CONFIG["event"]["a1"], "annual_parallax": False},
        "noise_model": "Known independent Gaussian errors from injected mean; photon/background/read approximation plus illustrative independent floor, no GP",
        "injected": data.truth,
        "sampled_parameter_names": list(LENS_NAMES),
        "posterior_output_order": list(NAMES),
        "flux_prior": {
            "order": ["f_s", "f_b"],
            "distribution": "Proper bivariate Gaussian",
            "mean": FLUX_PRIOR_MEAN.tolist(),
            "covariance": FLUX_PRIOR_COVARIANCE.tolist(),
            "scope": "Allows negative source/blend flux. This is a common explicit prior for all methods, not the former positive logit-fraction/log-baseline prior.",
        },
        "marginalization": "Normalized analytic Gaussian flux integral; sample four lens parameters, reconstruct paired conditional fluxes outside sampling timers",
        "priors_transformed": {
            "order": list(TRANSFORM_NAMES),
            "physical_output_order": list(LENS_NAMES),
            "location": PRIOR_LOC.tolist(),
            "scale": PRIOR_SCALE.tolist(),
            "density": "Independent Gaussian in stated coordinates, truncated to common rho/ratio domain; affine sampler transform has constant Jacobian",
            "common_domain": {"rho": [1e-6, 10.0], "max_u_over_rho": 1e8},
        },
    }
