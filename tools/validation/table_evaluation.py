"""Validate and time shared-radius contraction of the packaged table.

Run from the repository root with no competing numerical workload::

    JAX_PLATFORMS=cpu python -m tools.validation.table_evaluation

The reference performs the original full 4-by-4 coefficient gather at every
position. Both implementations use identical coefficients and cubic weights;
this measures an exact algebraic refactor, not a new physical approximation.
"""

import argparse
import hashlib
import json
import time
from importlib.resources import files
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import singlens as sl
from example._common import metadata
from singlens.lensing import BasisTable
from singlens.lensing.basis_table import _axis_weights

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "example/compare-single-vbml/outputs/reference_injection.npz"
EXPERIMENT = ROOT / "example/flux_marginalization/outputs/report.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FullGatherTable(BasisTable):
    """Original tensor-product contraction, retained only as a benchmark reference."""

    def _interpolate(self, radius2, rho):
        """Interpolate all stored channels once, with safe out-of-domain inputs."""
        rho = jnp.asarray(rho)
        if rho.ndim != 0:
            raise ValueError("rho must be scalar; use jax.vmap for multiple radii.")
        if jnp.iscomplexobj(rho):
            raise ValueError("rho must be real.")
        safe_rho = jnp.where(jnp.isfinite(rho) & (rho > 0), rho, self.rho_grid[0])
        ratio2 = radius2 / safe_rho**2
        valid = (
            (rho >= self.rho_grid[0])
            & (rho <= self.rho_grid[-1])
            & (ratio2 >= self.z_grid[0] ** 2)
            & (ratio2 <= self.z_grid[-1] ** 2)
        )
        # Strict comparisons preserve the interior one-sided spline derivative
        # at closed domain endpoints (clip would return half that derivative).
        x = jnp.log(safe_rho)
        x = jnp.where(x < self._knots_rho[3], self._knots_rho[3], x)
        x = jnp.where(x > self._knots_rho[-4], self._knots_rho[-4], x)
        if self.coordinate == "log1p_ratio2":
            y = jnp.log1p(ratio2)
        else:
            y = jnp.sqrt(jnp.where(radius2 == 0, 1.0, ratio2))
            y = jnp.where(radius2 == 0, 0.0, y)
        y = jnp.where(y < self._knots_z[3], self._knots_z[3], y)
        y = jnp.where(y > self._knots_z[-4], self._knots_z[-4], y)
        ix, wx = _axis_weights(x, self._knots_rho)
        iy, wy = _axis_weights(y, self._knots_z)
        c = self._coefficients[ix[:, None], iy[..., None, :]]
        result = jnp.sum(c * wx[:, None, None] * wy[..., None, :, None], axis=(-3, -2))
        return result, valid


def make_models():
    """Use identical physical coefficients with the original gather as reference."""
    with files("singlens.data").joinpath("finite_source_table.npz").open("rb") as stream:
        reference = FullGatherTable.load(stream)
    return {
        "shared_radius": sl.Model(source="linear"),
        "full_gather": sl.Model(source="linear", solver=reference),
    }


def equivalence(models):
    """Compare values and Cartesian/radius/limb-darkening derivatives."""
    ratios = np.unique(
        np.r_[
            0,
            np.linspace(0.01, 3, 256),
            0.99999,
            1,
            1.00001,
            np.geomspace(3, 1e8 * (1 - 1e-12), 256),
        ]
    )
    radii = np.geomspace(1e-6, 10, 9)
    coefficients = (0.0, 0.4, 1.0)
    positions = jnp.asarray(ratios * np.exp(1j * np.pi / 5))

    def vector(model, p, position):
        flux, shift = model.joint_at(position + p[0] + 1j * p[1], rho=p[2], a1=p[3])
        return jnp.stack((flux, shift.real, shift.imag), axis=-1)

    evaluate = {
        name: jax.jit(
            lambda p, z, m=model: (vector(m, p, z), jax.jacfwd(lambda q: vector(m, q, z))(p))
        )
        for name, model in models.items()
    }
    maxima, contact_maxima, off_contact_maxima = {}, {}, {}
    for rho in radii:
        for a1 in coefficients:
            p = jnp.array([0, 0, rho, a1])
            value, derivative = map(np.asarray, evaluate["shared_radius"](p, rho * positions))
            reference, reference_derivative = map(
                np.asarray, evaluate["full_gather"](p, rho * positions)
            )
            if not all(
                np.all(np.isfinite(x)) for x in (value, derivative, reference, reference_derivative)
            ):
                raise RuntimeError("Nonfinite value or derivative in equivalence check")
            difference = abs(derivative - reference_derivative) * np.array([rho, rho, rho, 1])
            errors = {
                "flux_relative": abs(value[:, 0] / reference[:, 0] - 1),
                "shift_absolute_theta_e": np.max(abs(value[:, 1:] - reference[:, 1:]), axis=1),
                "flux_gradient_scaled": np.max(difference[:, 0, :] / reference[:, 0, None], axis=1),
                "shift_gradient_scaled_theta_e": np.max(difference[:, 1:, :], axis=(1, 2)),
            }
            for metric, error in errors.items():
                for summary, selected in (
                    (maxima, np.ones(len(ratios), dtype=bool)),
                    (contact_maxima, ratios == 1),
                    (off_contact_maxima, ratios != 1),
                ):
                    index = int(np.argmax(np.where(selected, error, -1)))
                    if metric not in summary or error[index] > summary[metric]["error"]:
                        row = dict(
                            error=float(error[index]),
                            rho=float(rho),
                            ratio=float(ratios[index]),
                            a1=a1,
                        )
                        if "gradient" in metric:
                            if metric.startswith("flux"):
                                output, parameter = 0, int(np.argmax(difference[index, 0]))
                            else:
                                output, parameter = np.unravel_index(
                                    np.argmax(difference[index, 1:]), (2, 4)
                                )
                                output += 1
                            row.update(
                                output=("flux", "shift_x", "shift_y")[output],
                                parameter=("x", "y", "rho", "a1")[parameter],
                                derivative=float(derivative[index, output, parameter]),
                                reference_derivative=float(
                                    reference_derivative[index, output, parameter]
                                ),
                            )
                        summary[metric] = row
    # Contact knots are narrowly spaced: differentiating a rearranged floating
    # sum amplifies roundoff there, even though the polynomial is unchanged.
    tolerances = {
        "flux_relative": 1e-12,
        "shift_absolute_theta_e": 1e-12,
        "flux_gradient_scaled": 1e-9,
        "shift_gradient_scaled_theta_e": 1e-9,
    }
    if any(row["error"] > tolerances[metric] for metric, row in maxima.items()):
        raise RuntimeError(f"Contraction equivalence exceeded tolerance: {maxima}")
    return dict(
        scope="Algebraic equivalence to the original full-gather interpolator, including alignment/contact/wings; not independent physical-accuracy validation.",
        rho=radii.tolist(),
        u_over_rho=ratios.tolist(),
        a1=list(coefficients),
        parameters=["x", "y", "rho", "a1"],
        gradient_scaling="Multiply x, y, rho derivatives by rho; a1 by 1. Divide flux derivatives by magnification; shifts are in theta_E.",
        maxima=maxima,
        exact_contact_maxima=contact_maxima,
        off_contact_maxima=off_contact_maxima,
        contact_conditioning=_contact_conditioning(models, contact_maxima["flux_gradient_scaled"]),
        tolerances=tolerances,
        passed=True,
    )


def _contact_conditioning(models, worst):
    """Record identical spline inputs and the small spans amplifying roundoff."""
    rho = worst["rho"]
    position = jnp.asarray(rho * np.exp(1j * np.pi / 5))
    y = jnp.log1p((position.real**2 + position.imag**2) / rho**2)
    records = {}
    for name, model in models.items():
        table = model.solver
        ix, wx = _axis_weights(jnp.log(rho), table._knots_rho)
        iy, wy = _axis_weights(y, table._knots_z)
        span = int(iy[-1])
        records[name] = dict(
            radius_indices=np.asarray(ix).tolist(),
            radius_weights=np.asarray(wx).tolist(),
            separation_indices=np.asarray(iy).tolist(),
            separation_weights=np.asarray(wy).tolist(),
            neighboring_knot_spacings=np.diff(
                np.asarray(table._knots_z)[span - 3 : span + 5]
            ).tolist(),
        )
    if records["shared_radius"] != records["full_gather"]:
        raise RuntimeError("Spline indices/weights differ at the contact conditioning check")
    return dict(
        indices_and_weights_identical=True,
        spline_inputs=records["shared_radius"],
        explanation="Both formulations differentiate the same spline. Narrow contact knot spans amplify floating summation-order differences; maxima above retain the actual derivative amplitudes and separate exact-contact from off-contact cases.",
    )


def workloads():
    """Archived finite exposures plus explicit dense, wing, and scalar cases."""
    config = json.loads(EXPERIMENT.read_text())["experiment"]["physical_configuration"]
    truth = config["event"]
    with np.load(REFERENCE, allow_pickle=False) as data:
        times = np.asarray(data["times"])
    nodes, weights = np.polynomial.legendre.leggauss(3)
    offsets = nodes * config["exposure_seconds"] / (2 * 86400)
    rho, t_e, t0, u0 = (truth[key] for key in ("rho", "t_e", "t0", "u0"))
    cases = {
        "archived_season": (times[:, None] + offsets, weights / 2),
        "season_vmap16": (times[:, None] + offsets, weights / 2),
        "dense_transit": (t0 + np.linspace(-3, 3, 1000)[:, None] * rho * t_e, np.ones(1)),
        "far_wings": (t0 + np.geomspace(20, 1e4, 1000)[:, None] * rho * t_e, np.ones(1)),
        "scalar_transit": (np.array([[t0]]), np.ones(1)),
        "scalar_wing": (np.array([[t0 + 20 * rho * t_e]]), np.ones(1)),
    }
    details = {}
    for name, (subtimes, averaging) in cases.items():
        ratio = np.hypot((subtimes - t0) / t_e, u0) / rho
        details[name] = {
            "shape": list(subtimes.shape),
            "events": 16 if name == "season_vmap16" else 1,
            "epochs": len(subtimes),
            "evaluations": int(subtimes.size),
            "source_transit_positions_u_over_rho_lt_1": int(np.count_nonzero(ratio < 1)),
            "time_range_days": [float(subtimes.min()), float(subtimes.max())],
            "exposure_weights": averaging.tolist(),
            "subtimes_sha256": hashlib.sha256(subtimes.astype("<f8").tobytes()).hexdigest(),
        }
    return cases, truth, details


def timing(models, repetitions):
    """Alternate implementations; synchronize every value or value+VJP call."""
    cases, truth, descriptions = workloads()
    theta = jnp.array([truth[key] for key in ("t0", "u0", "t_e", "rho", "a1")])
    rows = []
    for case, (times, weights) in cases.items():
        times, weights = jnp.asarray(times), jnp.asarray(weights)
        for observable in ("photometry", "joint"):
            for operation in ("value", "value_and_gradient"):
                if case == "season_vmap16" and (observable != "photometry" or operation != "value"):
                    continue
                compiled = {}
                argument = (
                    jnp.tile(theta, (16, 1)).at[:, 0].add(jnp.linspace(-0.01, 0.01, 16))
                    if case == "season_vmap16"
                    else theta
                )
                for name, model in models.items():

                    def evaluate(p, m=model, joint=observable == "joint"):
                        params = sl.Parameters(t0=p[0], u0=p[1], t_e=p[2], rho=p[3], a1=p[4])
                        if joint:
                            flux, shift = m.joint(times, params)
                            return jnp.sum(flux * weights, axis=-1), jnp.sum(
                                shift * weights, axis=-1
                            )
                        return m.magnification(times, params) @ weights

                    if operation == "value_and_gradient":

                        def objective(p, function=evaluate, joint=observable == "joint"):
                            values = function(p)
                            if joint:
                                flux, shift = values
                                scalar = jnp.mean(flux + shift.real + 0.3 * shift.imag)
                            else:
                                scalar = jnp.mean(values)
                            return scalar, values

                        function = jax.value_and_grad(objective, has_aux=True)
                    else:
                        function = evaluate
                    if case == "season_vmap16":
                        function = jax.vmap(function)
                    compiled[name] = jax.jit(function)
                    for _ in range(3):
                        jax.block_until_ready(compiled[name](argument))
                measurements = {name: [] for name in models}
                names = tuple(models)
                for iteration in range(repetitions):
                    for name in names if iteration % 2 == 0 else names[::-1]:
                        start = time.perf_counter_ns()
                        jax.block_until_ready(compiled[name](argument))
                        measurements[name].append((time.perf_counter_ns() - start) / 1e9)
                medians = {name: float(np.median(values)) for name, values in measurements.items()}
                rows.append(
                    dict(
                        workload=case,
                        observable=observable,
                        operation=operation,
                        median_seconds=medians,
                        full_gather_over_shared_radius=medians["full_gather"]
                        / medians["shared_radius"],
                        seconds=measurements,
                    )
                )
                print(f"{case}: {observable}, {operation}: {medians}", flush=True)
    return {
        "scope": "Warm synchronized CPU calls, excluding construction and compilation. Value+gradient returns the full observable and its scalar-reduction reverse-mode gradient; this is not a likelihood, sampler, or ESS/s benchmark.",
        "comparison": "Identical physical table; shared-radius contraction versus the original per-position full coefficient gather.",
        "repetitions_per_method": repetitions,
        "order": "Alternating shared_radius/full_gather, then full_gather/shared_radius",
        "warmup_calls_per_executable": 3,
        "gradient_parameters": ["t0", "u0", "t_e", "rho", "a1"],
        "gradient_objective": "Mean magnification for photometry; mean(A + real(shift) + 0.3*imag(shift)) for joint; full values returned as auxiliary output.",
        "vmap16_scope": "Photometry values for 16 events, with t0 offsets evenly spaced over [-.01,.01] days; all other injected parameters shared.",
        "truth": truth,
        "workloads": descriptions,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "tools/validation/outputs/table_evaluation.json"
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("equivalence", "timing"),
        default=("equivalence", "timing"),
    )
    parser.add_argument("--repetitions", type=int, default=25)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    jax.config.update("jax_enable_x64", True)
    if jax.default_backend() != "cpu":
        parser.error("Use JAX_PLATFORMS=cpu for these timings")
    models = make_models()
    result = {
        "metadata": metadata(),
        "inputs_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (
                Path(__file__),
                REFERENCE,
                EXPERIMENT,
                ROOT / "src/singlens/data/finite_source_table.npz",
            )
        },
        "environment": {"backend": jax.default_backend(), "dtype": "float64"},
    }
    if "equivalence" in args.sections:
        result["equivalence"] = equivalence(models)
    if "timing" in args.sections:
        result["timing"] = timing(models, args.repetitions)
    for relative, checksum in result["metadata"]["source_sha256"].items():
        if _sha256(Path(sl.__file__).parent / relative) != checksum:
            raise RuntimeError(f"Source changed during validation: {relative}")
    for relative, checksum in result["inputs_sha256"].items():
        if _sha256(ROOT / relative) != checksum:
            raise RuntimeError(f"Input changed during validation: {relative}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
