"""Synthetic stellar-lens photometry with finite-source and annual-parallax signals.

The event resembles OGLE-2007-BLG-050; the cadence, detector, and noise are
explicit illustrative assumptions, not the observations of that event.
"""

from __future__ import annotations

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

from example.parallax_inference.code.reference import outside_integral, outside_rule
from singlens import AnnualParallax, Model

LENS_NAMES = ("t0", "u0", "t_e", "rho", "pi_e_n", "pi_e_e", "a1")
NAMES = (*LENS_NAMES, "f_s", "f_b")
LABELS = (
    r"$t_0$ (day)",
    r"$u_0$",
    r"$t_E$ (day)",
    r"$\rho$",
    r"$\pi_{E,N}$",
    r"$\pi_{E,E}$",
    r"$a_1$",
    r"$f_s$",
    r"$f_b$",
)
TRANSFORM_NAMES = (
    "t0",
    "log_abs_u0_over_rho",
    "log_source_crossing_time",
    "log_rho",
    "pi_e_n",
    "pi_e_e",
    "logit_a1",
)
# Densities in these coordinates, conditional on a specified impact-parameter sign.
PRIOR_LOC = np.array([0.0, np.log(0.5), np.log(0.3), np.log(0.005), 0.0, 0.0, 0.0])
PRIOR_SCALE = np.array([0.2, 0.7, 0.5, 0.7, 0.3, 0.3, 1.0])
FLUX_PRIOR_MEAN = np.array([0.8, 0.2])
FLUX_PRIOR_COVARIANCE = np.array([[0.0272, -0.0252], [-0.0252, 0.0257]])
EXPOSURE_ORDER, INJECTION_ORDER, DATA_SEED = 3, 9, 43


def configuration():
    theta_e, theta_star, pi_e, distance, t_e = 0.48, 0.0022, 0.12, 8.0, 67.0
    rho, impact_ratio = theta_star / theta_e, 0.46
    source_mag, source_fraction, zero_point = 18.5, 0.8, 25.0
    source_rate = 10 ** (-0.4 * (source_mag - zero_point))
    return dict(
        description="Synthetic finite-source stellar event inspired by OGLE-2007-BLG-050",
        event=dict(
            t0=0.0,
            u0=impact_ratio * rho,
            t_e=t_e,
            rho=rho,
            pi_e_n=0.096,
            pi_e_e=-0.072,
            a1=0.5,
            f_s=source_fraction,
            f_b=1 - source_fraction,
        ),
        theta_e_mas=theta_e,
        theta_star_mas=theta_star,
        pi_e=pi_e,
        pi_rel_mas=theta_e * pi_e,
        lens_mass_solar=theta_e / (8.144 * pi_e),
        source_distance_kpc=distance,
        lens_distance_kpc=1 / (1 / distance + theta_e * pi_e),
        source_radius_solar=theta_star
        / 1000
        / 206264.80624709636
        * distance
        * 1000
        * 3.085677581491367e16
        / 6.957e8,
        mu_geo_masyr=theta_e / t_e * 365.25,
        source_crossing_hours=rho * t_e * 24,
        transit_hours=2 * rho * t_e * np.sqrt(1 - impact_ratio**2) * 24,
        parallax=dict(ra_deg=266.0, dec_deg=-29.0, t_ref=0.0, time_origin=2454222.0, use_hjd=False),
        observer="Earth geocenter; the bundled heliocentric Earth ephemeris, without terrestrial parallax",
        time_convention="JD TDB minus 2454222.0",
        observing_scope="Idealized longitude-distributed ground network over one bulge season, with continuous peak follow-up; weather and site visibility are not simulated",
        cadence=dict(
            first_day=-60,
            last_day=150,
            wing_days=1,
            shoulder_hours=1,
            shoulder_half_width_days=10,
            peak_minutes=5,
            peak_half_width_days=1,
        ),
        exposure_scope="Fixed pre-injection point-source forecast: clip(60/A_forecast, 1, 60) seconds; top-hat averages",
        source_magnitude=source_mag,
        baseline_magnitude=source_mag + 2.5 * np.log10(source_fraction),
        band="Illustrative ground I band",
        source_fraction=source_fraction,
        instrument_zero_point_mag_per_electron_second=zero_point,
        source_electrons_per_second=source_rate,
        baseline_electrons_per_second=source_rate / source_fraction,
        aperture_sky_electrons_per_second=500.0,
        aperture_read_variance_electrons2=400.0,
        fractional_floor=0.003,
        representative_peak_pixel_fraction=0.1,
        fullwell_electrons=100000.0,
        detector_scope="Illustrative detector and seeing-spread PSF; not an instrument ETC or survey forecast. Apparent source brightness is specified independently of its angular radius",
        noise_scope="Known independent Gaussian errors from injected source+blend photons, sky, aperture read variance, and an independent 0.3% fractional floor; recomputed for each scheduled exposure",
        blend_scope="Unresolved constant light; source and blend coefficients share a normalized Gaussian prior and are sampled jointly",
        limb_darkening="Inferred linear a1 with logit(a1)~Normal(0,1); injected a1=0.5 is an illustrative I-band profile. The actual inspiration event also fitted limb darkening (Batista et al. 2009, Section 3.1); its Gamma coefficient uses a different convention",
        references={
            "event": "https://doi.org/10.1051/0004-6361/200912923",
            "event_preprint": "https://arxiv.org/abs/0907.3471",
        },
    )


CONFIG = configuration()


class Geometry(NamedTuple):
    times: jax.Array
    weights: jax.Array
    tau_n: jax.Array
    beta_n: jax.Array
    tau_e: jax.Array
    beta_e: jax.Array


class Injection(NamedTuple):
    times: jax.Array
    flux: jax.Array
    error: jax.Array
    truth: dict
    lens_flux: jax.Array
    exposure_seconds: jax.Array
    geometry: Geometry
    seed: int = DATA_SEED
    reference_checks: dict | None = None


@cache
def parallax_geometry():
    return AnnualParallax(**CONFIG["parallax"])


@cache
def exposure_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return nodes / (2 * 86400), weights / 2


def prepare_geometry(times, exposure_seconds, *, order=EXPOSURE_ORDER):
    """Precompute the two linear parallax bases outside every likelihood call."""
    offsets, weights = exposure_rule(order)
    times = jnp.asarray(times)[:, None] + jnp.asarray(exposure_seconds)[:, None] * offsets
    geometry = parallax_geometry()
    tau_n, beta_n = geometry.offsets(times, pi_e_n=1.0, pi_e_e=0.0)
    tau_e, beta_e = geometry.offsets(times, pi_e_n=0.0, pi_e_e=1.0)
    return Geometry(times, jnp.asarray(weights), tau_n, beta_n, tau_e, beta_e)


def observation_times():
    """1199 epochs: daily wings, hourly shoulders, and five-minute peak coverage."""
    wings = np.r_[np.arange(-60, -10), np.arange(11, 151)]
    shoulders = np.r_[np.arange(-240, -24), np.arange(25, 241)] / 24
    peak = np.arange(-288, 289) / 288
    return np.sort(np.r_[wings, shoulders, peak])


def exposure_schedule(times):
    """One fixed observing schedule, independent of all fitted parameters."""
    forecast = CONFIG["event"]
    u = np.hypot((np.asarray(times) - forecast["t0"]) / forecast["t_e"], forecast["u0"])
    amplitude = (u**2 + 2) / (u * np.hypot(u, 2))
    return np.clip(60 / amplitude, 1, 60)


def unpack(theta, sign=1):
    rho, t_star = jnp.exp(theta[3]), jnp.exp(theta[2])
    return dict(
        t0=theta[0],
        u0=sign * rho * jnp.exp(theta[1]),
        t_e=t_star / rho,
        rho=rho,
        pi_e_n=theta[4],
        pi_e_e=theta[5],
        a1=jax.nn.sigmoid(theta[6]),
    )


def pack(params):
    """Coordinates for either sign branch; source and blend flux are omitted."""
    return np.array(
        [
            params["t0"],
            np.log(abs(params["u0"]) / params["rho"]),
            np.log(params["rho"] * params["t_e"]),
            np.log(params["rho"]),
            params["pi_e_n"],
            params["pi_e_e"],
            np.log(params["a1"] / (1 - params["a1"])),
        ]
    )


def position(geometry, params):
    tau = (geometry.times - params["t0"]) / params["t_e"]
    tau = tau + params["pi_e_n"] * geometry.tau_n + params["pi_e_e"] * geometry.tau_e
    beta = params["u0"] + params["pi_e_n"] * geometry.beta_n + params["pi_e_e"] * geometry.beta_e
    return tau + 1j * beta


def make_model(solver="table"):
    return Model(source="linear", solver=solver)


def magnification_parameters(model, data, params, *, geometry=None):
    geometry = data.geometry if geometry is None else geometry
    values = model.magnification_at(position(geometry, params), rho=params["rho"], a1=params["a1"])
    return jnp.sum(values * geometry.weights, axis=-1)


def magnification(model, data, theta, sign=1, *, geometry=None):
    return magnification_parameters(model, data, unpack(theta, sign), geometry=geometry)


def log_prior(theta):
    valid = jnp.all(jnp.isfinite(theta)) & (theta[3] >= np.log(1e-6)) & (theta[3] <= np.log(10))
    density = -0.5 * jnp.sum(((theta - PRIOR_LOC) / PRIOR_SCALE) ** 2)
    return jnp.where(valid, density, -jnp.inf)


def photometric_error(mean_flux, exposure_seconds):
    mean, exposure = np.broadcast_arrays(np.asarray(mean_flux), np.asarray(exposure_seconds))
    if (
        np.any(~np.isfinite(mean))
        or np.any(mean <= 0)
        or np.any(~np.isfinite(exposure))
        or np.any(exposure <= 0)
    ):
        raise ValueError("Flux and exposure must be finite and positive")
    counts = CONFIG["baseline_electrons_per_second"] * exposure
    variance = (
        mean * counts
        + CONFIG["aperture_sky_electrons_per_second"] * exposure
        + CONFIG["aperture_read_variance_electrons2"]
    )
    return np.sqrt(variance / counts**2 + (CONFIG["fractional_floor"] * mean) ** 2)


def saturation_check(mean_flux, exposure_seconds):
    mean, exposure = np.broadcast_arrays(np.asarray(mean_flux), np.asarray(exposure_seconds))
    pixels = (
        mean
        * CONFIG["baseline_electrons_per_second"]
        * exposure
        * CONFIG["representative_peak_pixel_fraction"]
    )
    return dict(
        brightest_magnitude=float(CONFIG["baseline_magnitude"] - 2.5 * np.log10(np.max(mean))),
        maximum_peak_pixel_electrons=float(np.max(pixels)),
        fullwell_electrons=CONFIG["fullwell_electrons"],
        passed=bool(
            np.all(np.isfinite(pixels))
            and np.all(pixels > 0)
            and np.max(pixels) < CONFIG["fullwell_electrons"]
        ),
        scope="Illustrative detector, fixed scheduled exposures, and seeing-spread peak pixel; excludes sky saturation",
    )


def reference_magnification(
    times, exposure_seconds, params, *, order=INJECTION_ORDER, spatial_order=32
):
    """Independent source quadrature with the same prepared Earth trajectory."""
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    wings = import_module("example.compare-single-vbml.code.experiment")._wing_integral
    geometry = prepare_geometry(times, exposure_seconds, order=order)
    u = np.abs(np.asarray(position(geometry, params)))
    flat, rho, a1 = u.ravel(), float(params["rho"]), float(params["a1"])
    ratio = flat / rho
    near = ratio < 1.05
    outer = (ratio >= 1.05) & (ratio <= 3)
    far = ratio > 3
    values = np.empty_like(flat)
    values[near] = [integral(float(x), rho, a1=a1, tolerance=1e-11) for x in flat[near]]
    values[outer] = outside_integral(flat[outer], rho, a1=a1, order=2 * spatial_order)
    values[far] = wings(flat[far], rho, a1, spatial_order)
    return np.sum(values.reshape(u.shape) * np.asarray(geometry.weights), axis=-1)


def reference_provenance():
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    wings = import_module("example.compare-single-vbml.code.experiment")._wing_integral
    # The bundled ephemeris is fingerprinted through the actual prepared arrays
    # below; this also records changes to the projection/time convention.
    times = observation_times()
    prepared = prepare_geometry(times, exposure_schedule(times), order=INJECTION_ORDER)
    geometry_hash = hashlib.sha256(
        b"".join(np.asarray(x, dtype="<f8").tobytes() for x in prepared)
    ).hexdigest()
    return dict(
        source_sha256={
            f.__name__: hashlib.sha256(inspect.getsource(f).encode()).hexdigest()
            for f in (
                integral,
                wings,
                outside_integral,
                outside_rule,
                reference_magnification,
                prepare_geometry,
                position,
                exposure_rule,
                observation_times,
                exposure_schedule,
                photometric_error,
                saturation_check,
            )
        },
        configuration_sha256=hashlib.sha256(
            json.dumps(CONFIG, sort_keys=True).encode()
        ).hexdigest(),
        prepared_injection_geometry_sha256=geometry_hash,
        fit_exposure_order=EXPOSURE_ORDER,
        injection_exposure_order=INJECTION_ORDER,
    )


def reference_data_hash(times, exposure, error, mean):
    return hashlib.sha256(
        np.column_stack([times, exposure, error, mean]).astype("<f8").tobytes()
    ).hexdigest()


def _validate_reference_data(times, exposure, error, mean, checks):
    arrays = tuple(np.asarray(value) for value in (times, exposure, error, mean))
    if (
        any(
            value.ndim != 1 or value.size == 0 or not np.all(np.isfinite(value)) for value in arrays
        )
        or len({value.shape for value in arrays}) != 1
    ):
        raise ValueError("Reference arrays must be finite nonempty matching vectors")
    if any(np.any(value <= 0) for value in arrays[1:]):
        raise ValueError("Reference exposure, errors, and mean must be positive")
    if not np.array_equal(times, observation_times()) or not np.array_equal(
        exposure, exposure_schedule(times)
    ):
        raise ValueError("Reference observing schedule differs")
    if checks.get("saturation", {}).get("passed") is not True:
        raise ValueError("Reference saturation validation failed")
    for name, tolerance in (
        ("injection_refinement_max_flux_error_sigma", 1e-3),
        ("fitting_exposure_rule_max_flux_error_sigma", 0.01),
        ("overlap_max_relative_difference", 1e-9),
    ):
        value = checks.get(name, np.nan)
        if not np.isfinite(value) or not 0 <= value <= tolerance:
            raise ValueError(f"Reference validation failed: {name}")


def save_reference_data(path, *, times, exposure, error, mean, checks):
    _validate_reference_data(times, exposure, error, mean, checks)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    description = dict(
        provenance=reference_provenance(),
        checks=checks,
        numeric_payload_sha256=reference_data_hash(times, exposure, error, mean),
    )
    np.savez_compressed(
        path,
        times=times,
        exposure=exposure,
        error=error,
        mean=mean,
        metadata=json.dumps(description, sort_keys=True),
    )


def load_reference_data(path):
    with np.load(path, allow_pickle=False) as saved:
        times, exposure, error, mean = (
            saved[name].copy() for name in ("times", "exposure", "error", "mean")
        )
        description = json.loads(str(saved["metadata"]))
    if description["provenance"] != reference_provenance():
        raise ValueError("Reference configuration or numerical source hashes differ")
    _validate_reference_data(times, exposure, error, mean, description["checks"])
    if description["numeric_payload_sha256"] != reference_data_hash(times, exposure, error, mean):
        raise ValueError("Reference payload checksum mismatch")
    return times, exposure, error, mean, description["checks"]


def synthetic_data(*, seed=DATA_SEED, reference=None):
    truth = {name: float(CONFIG["event"][name]) for name in NAMES}
    if reference is not None and Path(reference).exists():
        times, exposure, error, mean, checks = load_reference_data(reference)
    else:
        times = observation_times()
        exposure = exposure_schedule(times)
        print("Independent injection: 9 exposure nodes and exact source quadrature.", flush=True)
        values = reference_magnification(times, exposure, truth)
        mean = truth["f_s"] * values + truth["f_b"]
        error = photometric_error(mean, exposure)
        # Refine all source-sensitive epochs plus representative outer wings.
        u = np.abs(np.asarray(position(prepare_geometry(times, exposure, order=1), truth)))[:, 0]
        selected = np.unique(
            np.r_[
                np.flatnonzero(u < 3 * truth["rho"]), np.linspace(0, len(times) - 1, 32, dtype=int)
            ]
        )
        print(
            f"Independent refinement: {len(selected)} epochs, 15 exposure nodes and doubled spatial rules.",
            flush=True,
        )
        refined = reference_magnification(
            times[selected], exposure[selected], truth, order=15, spatial_order=64
        )
        print("Checking the 3-node fitting exposure rule.", flush=True)
        coarse = reference_magnification(
            times[selected], exposure[selected], truth, order=EXPOSURE_ORDER
        )
        scale = truth["f_s"] / error[selected]
        integral = import_module("example.compare-single-vbml.code.accuracy").integral
        wings = import_module("example.compare-single-vbml.code.experiment")._wing_integral
        overlap_u = truth["rho"] * np.array([2.8, 3.0, 3.2])
        overlap_reference = np.array(
            [integral(x, truth["rho"], a1=truth["a1"], tolerance=1e-11) for x in overlap_u]
        )
        overlap = wings(overlap_u, truth["rho"], truth["a1"], 32)
        checks = dict(
            saturation=saturation_check(mean, exposure),
            selected_epochs=selected.tolist(),
            injection_refinement_max_flux_error_sigma=float(
                np.max(np.abs((values[selected] - refined) * scale))
            ),
            fitting_exposure_rule_max_flux_error_sigma=float(
                np.max(np.abs((coarse - refined) * scale))
            ),
            overlap_max_relative_difference=float(np.max(np.abs(overlap / overlap_reference - 1))),
            injection_order=INJECTION_ORDER,
            refined_exposure_order=15,
            wing_orders=[32, 64],
            transformed_outside_orders=[64, 128],
            transformed_outside_range_u_over_rho=[1.05, 3.0],
            wing_boundary_u_over_rho=3,
            epochs_inside_source=int(np.sum(u < truth["rho"])),
            epochs_within_three_source_radii=int(np.sum(u < 3 * truth["rho"])),
        )
        _validate_reference_data(times, exposure, error, mean, checks)
        if reference is not None:
            save_reference_data(
                reference, times=times, exposure=exposure, error=error, mean=mean, checks=checks
            )
    flux = mean + np.random.default_rng(seed).normal(size=len(times)) * error
    return Injection(
        jnp.asarray(times),
        jnp.asarray(flux),
        jnp.asarray(error),
        truth,
        jnp.asarray(mean),
        jnp.asarray(exposure),
        prepare_geometry(times, exposure),
        seed,
        checks,
    )


def data_hash(data):
    return hashlib.sha256(
        np.column_stack([data.times, data.exposure_seconds, data.flux, data.error, data.lens_flux])
        .astype("<f8")
        .tobytes()
    ).hexdigest()


def _validation_epochs(data, params):
    """Include the peak, both contacts, and outer wings in a fixed-size subset."""
    geometry = prepare_geometry(data.times, data.exposure_seconds, order=1)
    ratio = np.abs(np.asarray(position(geometry, params)))[:, 0] / float(params["rho"])
    times = np.asarray(data.times)
    selected = {int(np.argmin(ratio))}
    for before in (True, False):
        candidates = np.flatnonzero((times < float(params["t0"])) == before)
        if not len(candidates):
            continue
        for target in (0.5, 0.9, 1.0, 1.1, 2.0, 3.0):
            index = int(candidates[np.argmin(np.abs(ratio[candidates] - target))])
            selected.add(index)
            if target == 1.0:
                selected.update((max(0, index - 1), min(len(times) - 1, index + 1)))
    selected.update(np.linspace(0, len(times) - 1, 8, dtype=int))
    count = min(25, len(times))
    for index in np.linspace(0, len(times) - 1, len(times), dtype=int):
        if len(selected) >= count:
            break
        selected.add(int(index))
    return np.asarray(sorted(selected), dtype=int)


def validate_model(model, data, states):
    """Check independent values and exposure integration before using timings.

    ``states`` contains physical parameter dictionaries, with truth and MAP first.
    Independent source quadrature checks those first two states. Exposure
    refinement checks every supplied state, typically including twelve posterior
    draws. All discrepancies are measured against the recorded per-epoch noise.
    """
    if not states:
        raise ValueError("Supply at least one physical parameter state")
    evaluate = jax.jit(
        lambda params, geometry: magnification_parameters(model, data, params, geometry=geometry)
    )
    independent, exposures = [], []
    for index, state in enumerate(states):
        params = {name: float(state[name]) for name in LENS_NAMES}
        selected = _validation_epochs(data, params)
        times = np.asarray(data.times)[selected]
        duration = np.asarray(data.exposure_seconds)[selected]
        errors = np.asarray(data.error)[selected]
        geometries = [prepare_geometry(times, duration, order=order) for order in (3, 9, 15)]
        coarse, middle, fine = (np.asarray(evaluate(params, item)) for item in geometries)
        source_flux = float(state["f_s"])
        scale = abs(source_flux) / errors
        discrepancy = np.abs((coarse - fine) * scale)
        refinement = np.abs((middle - fine) * scale)
        exposures.append(
            dict(
                state=index,
                epochs=selected.tolist(),
                source_flux=source_flux,
                max_abs_three_minus_fifteen_sigma=float(np.max(discrepancy)),
                max_abs_nine_minus_fifteen_sigma=float(np.max(refinement)),
            )
        )
        if index < 2:
            reference = reference_magnification(times, duration, params, order=15, spatial_order=64)
            check = reference_magnification(times, duration, params, order=9, spatial_order=32)
            independent.append(
                dict(
                    state=index,
                    epochs=selected.tolist(),
                    max_abs_fit_minus_independent_sigma=float(
                        np.max(np.abs((coarse - reference) * scale))
                    ),
                    max_abs_reference_refinement_sigma=float(
                        np.max(np.abs((check - reference) * scale))
                    ),
                    max_relative_magnification_difference=float(
                        np.max(np.abs(coarse / reference - 1))
                    ),
                )
            )
    maximum_error = max(item["max_abs_fit_minus_independent_sigma"] for item in independent)
    maximum_reference = max(item["max_abs_reference_refinement_sigma"] for item in independent)
    maximum_exposure = max(item["max_abs_three_minus_fifteen_sigma"] for item in exposures)
    passed = bool(
        np.isfinite([maximum_error, maximum_reference, maximum_exposure]).all()
        and maximum_error < 0.01
        and maximum_reference < 0.001
        and maximum_exposure < 0.01
    )
    return dict(
        passed=passed,
        scope="Peak, both contacts and wings at 25 observed epochs per state. Independent spatial quadrature checks the first two supplied states (truth and MAP); 3/9/15-node exposure averaging checks every supplied state. This is a sampled validation, not a uniform error bound.",
        threshold_sigma=0.01,
        reference_refinement_threshold_sigma=0.001,
        max_abs_fit_minus_independent_sigma=maximum_error,
        max_abs_reference_refinement_sigma=maximum_reference,
        max_abs_three_minus_fifteen_sigma=maximum_exposure,
        independent=independent,
        exposure_refinement=exposures,
    )
