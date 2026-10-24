"""Check the rectilinear approximation for the physical photometry benchmark.

This is an Earth-geocentric sensitivity calculation at fixed event parameters,
not a Roman ephemeris forecast or a posterior-bias calculation. Run separately
from timing measurements. An existing inference_event.csv avoids regenerating
the independent brightness-integration injection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib import import_module, resources
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from example._common import model_solver_metadata
from example._physical import photometric_error
from singlens import AnnualParallax, Model, Parameters

experiment = import_module("example.compare-single-vbml.code.experiment")
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "physical_checks.json"
RA_DEG, DEC_DEG = 268.07, -29.23
TIME_ORIGINS = 2462030.0 + np.arange(4) * 365.25 / 4
DIRECTIONS = np.arange(8) * np.pi / 4


def load_mean_and_errors(event_csv=None):
    """Use only the common expected flux and known errors, not the noise draw."""
    if event_csv is None:
        data = experiment.synthetic_data(seed=experiment.DATA_SEED)
        times, mean, error = map(np.asarray, (data.times, data.lens_flux, data.error))
        provenance = dict(
            origin="Regenerated independent injection", reference_checks=data.reference_checks
        )
    else:
        event_csv = Path(event_csv)
        saved = np.loadtxt(event_csv, delimiter=",", skiprows=1)
        if saved.ndim != 2 or saved.shape[1] != 4:
            raise ValueError("Expected inference_event.csv columns: time, flux, error, mean")
        times, mean, error = saved[:, 0], saved[:, 3], saved[:, 2]
        np.testing.assert_allclose(times, experiment.observation_times(), rtol=0, atol=1e-12)
        provenance = dict(
            origin="Saved independent-injection event",
            file=event_csv.name,
            file_sha256=hashlib.sha256(event_csv.read_bytes()).hexdigest(),
        )
    if not all(np.isfinite(value).all() for value in (times, mean, error)):
        raise ValueError("Expected flux, errors and times must be finite")
    if np.any(mean <= 0) or np.any(error <= 0):
        raise ValueError("Expected flux and errors must be positive")
    np.testing.assert_allclose(error, photometric_error(mean, experiment.CONFIG), rtol=1e-12)
    values = np.column_stack([times, mean, error]).astype("<f8")
    provenance["time_mean_error_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    return times, mean, error, provenance


def parallax_omission_check(times, mean, error):
    """Compare exposure-averaged parallax and rectilinear curves at common truth."""
    config = experiment.CONFIG
    truth = {name: float(config["event"][name]) for name in experiment.NAMES}
    model = experiment.make_model("table")
    _, weights = experiment.exposure_rule(experiment.EXPOSURE_ORDER)
    subtimes = experiment.exposure_times(times)
    rectilinear = np.asarray(experiment.magnification(model, times, truth))
    baseline = truth["f_s"] * rectilinear + truth["f_b"]
    table_difference = (baseline - mean) / error
    rows = []
    for origin in TIME_ORIGINS:
        parallax = AnnualParallax(
            ra_deg=RA_DEG,
            dec_deg=DEC_DEG,
            t_ref=truth["t0"],
            time_origin=float(origin),
        )
        curved = Model(source="linear", solver="table", parallax=parallax)

        def evaluate(pi_n, pi_e):
            event = Parameters(
                **{name: truth[name] for name in ("t0", "u0", "t_e", "rho")},
                a1=config["event"]["a1"],
                pi_e_n=pi_n,
                pi_e_e=pi_e,
            )
            return jnp.sum(curved.magnification(subtimes, event) * weights, axis=-1)

        evaluate = jax.jit(evaluate)
        for angle in DIRECTIONS:
            pi_n, pi_e = config["pi_e"] * np.array([np.cos(angle), np.sin(angle)])
            values = np.asarray(evaluate(pi_n, pi_e))
            difference = truth["f_s"] * (values - rectilinear) / error
            if not np.isfinite(difference).all():
                raise ValueError("Nonfinite parallax sensitivity result")
            rows.append(
                dict(
                    time_origin=float(origin),
                    direction_radian=float(angle),
                    pi_e_n=float(pi_n),
                    pi_e_e=float(pi_e),
                    max_abs_flux_difference_sigma=float(np.max(np.abs(difference))),
                    rms_flux_difference_sigma=float(np.sqrt(np.mean(difference**2))),
                    sum_squared_flux_difference_sigma=float(np.sum(difference**2)),
                )
            )
    return dict(
        scope="Earth-geocentric annual-parallax perturbations at fixed injected parameters, four quarterly origins and eight orientations. Differences use identical exposure averaging and common known errors. No refitting, posterior bias, or branch evidence is inferred; this discrete grid is not a global bound. No Roman L2 orbit or survey visibility is modeled.",
        convention="Parallax mean minus rectilinear mean; pi_E,N=pi_E*cos(angle), pi_E,E=pi_E*sin(angle). Both curves use the bundled table so numerical lens errors largely cancel.",
        ra_deg=RA_DEG,
        dec_deg=DEC_DEG,
        time_convention="HJD TDB = relative time + time_origin; AnnualParallax(use_hjd=True)",
        t_ref=truth["t0"],
        pi_e=config["pi_e"],
        time_origins=TIME_ORIGINS.tolist(),
        directions_radian=DIRECTIONS.tolist(),
        exposure_order=experiment.EXPOSURE_ORDER,
        exposure_seconds=config["exposure_seconds"],
        observations=len(times),
        fit_solver=model_solver_metadata(model),
        rectilinear_table_vs_independent_injection=dict(
            max_abs_flux_difference_sigma=float(np.max(np.abs(table_difference))),
            sum_squared_flux_difference_sigma=float(np.sum(table_difference**2)),
        ),
        maximum_over_cases=dict(
            max_abs_flux_difference_sigma=max(row["max_abs_flux_difference_sigma"] for row in rows),
            sum_squared_flux_difference_sigma=max(
                row["sum_squared_flux_difference_sigma"] for row in rows
            ),
        ),
        likelihood_interpretation="At fixed parameters and equal known Gaussian variances, half the sum of squared standardized mean differences is the expected log-likelihood loss. It is not the realized loss of a particular noise draw or a fitted detection significance.",
        cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-csv", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    times, mean, error, provenance = load_mean_and_errors(args.event_csv)
    report = dict(
        physical_configuration=experiment.CONFIG,
        input=provenance,
        earth_ephemeris_sha256=hashlib.sha256(
            resources.files("singlens.data")
            .joinpath("earth_orbital_parallax_table.txt")
            .read_bytes()
        ).hexdigest(),
        parallax_omission=parallax_omission_check(times, mean, error),
        source_sha256={
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(experiment.__file__),
                Path(__file__).parents[2] / "_physical.py",
            )
        },
        environment=dict(
            jax=jax.__version__, numpy=np.__version__, x64_enabled=jax.config.x64_enabled
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["parallax_omission"]["maximum_over_cases"], indent=2))


if __name__ == "__main__":
    main()
