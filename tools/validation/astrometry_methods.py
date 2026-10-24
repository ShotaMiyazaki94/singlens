"""Independent astrometric accuracy, derivatives, throughput, and joint NUTS.

Examples (run from the repository root)::

    python -m tools.validation.astrometry_methods --sections accuracy gradients
    python -m tools.validation.astrometry_methods --with-tables --nuts

All coordinates and shifts are in Einstein-radius units unless explicitly
multiplied by theta_E in the synthetic astrometric observations. The reference
integrates image flux and image first moments directly, independently of the
concentric-disk and FFT implementations. Its weak-wing form avoids subtraction
of a large source position from a nearly equal image centroid.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")

from example._common import metadata
from tools.validation import astrometry_accuracy as independent
from tools.validation.finite_source_methods import (
    ReferenceCache as PhotometryReference,
)
from tools.validation.finite_source_methods import (
    clean_json,
    synchronize,
)

OUTPUT = Path(__file__).parent / "outputs"
RADII = (1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)
COEFFICIENTS = (0.0, 0.5, 1.0)
NUTS_TIMING_SCOPE = (
    "Sampling includes compilation of the post-warmup sampling executable. Warmup+sampling "
    "totals include their compilation, but exclude solver construction, offline table building, "
    "initial truth-model evaluation, reference generation, and independent likelihood validation. "
    "These are not cold-process end-to-end timings."
)
NUTS_COMPARISON_SCOPE = (
    "Posterior differences versus disk_64_32 use an independent-mean combined-MCSE heuristic. "
    "Methods use identical RNG seeds, so cross-method Monte Carlo covariance is unknown; these "
    "ratios are descriptive, not formal significance tests or numerical-bias bounds."
)
NUTS_ASTRO_ERROR_SCOPE = (
    "truth_max_astro_error_sigma is Euclidean two-dimensional model error divided by nominal "
    "sigma_x. The independent likelihood checks and numerical_model_bias_mahalanobis_squared "
    "use the full anisotropic, correlated covariance."
)


class Reference:
    """Cache independent uniform/pure-mu fluxes and normalized shifts."""

    def __init__(self):
        self.path = OUTPUT / "astrometry_methods_reference.json"
        self.photometry = PhotometryReference(OUTPUT / "finite_source_reference.json")
        signature = independent.reference_signature() + self.photometry.signature
        self.signature = hashlib.sha256(signature.encode()).hexdigest()
        saved = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.values = saved.get("values", {}) if saved.get("signature") == self.signature else {}
        old_path = OUTPUT / "astrometry_reference.json"
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        self.old = (
            old.get("values", {})
            if old.get("signature") == independent.reference_signature()
            else {}
        )

    def basis(self, u, rho):
        u, rho = abs(float(u)), float(rho)
        key = f"{u.hex()}:{rho.hex()}"
        if key not in self.values:
            flux = self.photometry.basis(u, rho)
            shifts = []
            for profile, a1 in (("uniform", None), ("linear", 1.0)):
                old_key = f"{profile}:{a1}:{rho:.16g}:{u:.16g}"
                shift = self.old.get(old_key)
                if shift is None:
                    shift = independent.reference_shift(u, rho, profile, a1)
                shifts.append(float(shift))
            self.values[key] = [*flux.tolist(), *shifts]
        return np.asarray(self.values[key])

    def joint(self, position, rho, a1):
        u = abs(position)
        f0, f1, s0, s1 = self.basis(u, rho)
        gamma = 2 * a1 / (3 - a1)
        flux = (1 - gamma) * f0 + gamma * f1
        shift = ((1 - gamma) * f0 * s0 + gamma * f1 * s1) / flux
        return float(flux), complex(shift * position / u) if u else 0j

    def vector(self, parameters):
        x, y, rho, a1 = parameters
        flux, shift = self.joint(complex(x, y), rho, a1)
        return np.array([flux, shift.real, shift.imag])

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(signature=self.signature, values=self.values)) + "\n")
        self.photometry.save()


class FFTJoint:
    """The existing photometric and astrometric solvers, evaluated together."""

    def __init__(self, n_fft):
        from singlens.lensing import FFTConfig, LinearLimbDarkened
        from singlens.lensing.astrometry import FiniteSourceAstrometry

        self.profile = LinearLimbDarkened(fft=FFTConfig(n_fft=n_fft))
        self.astrometry = FiniteSourceAstrometry(self.profile)

    def joint(self, position, rho, *, a1=0.5):
        p = jnp.asarray(position)
        r2 = p.real**2 + p.imag**2
        radius = jnp.where(r2 == 0, 0.0, jnp.sqrt(jnp.where(r2 == 0, 1.0, r2)))
        return self.profile.A(radius, rho, a1=a1), self.astrometry.shift(p, rho, a1=a1)

    def shift(self, position, rho, *, a1=0.5):
        return self.astrometry.shift(position, rho, a1=a1)


def make_methods(args):
    from singlens.lensing.concentric import ConcentricDisk

    available = {f"fft_{n}": FFTJoint(n) for n in (4096, 8192)}
    available.update(
        {
            f"disk_{n}_{m}": ConcentricDisk(n_disk=n, n_limb=m)
            for n, m in ((32, 16), (64, 32), (96, 64))
        }
    )
    tables = []
    if args.with_tables:
        from singlens.lensing.basis_table import BasisTable, joint_grid

        for n_rho, refinement in ((96, 1), (192, 2)):
            if args.table_sizes and n_rho not in args.table_sizes:
                continue
            knots = joint_grid()
            fractions = np.arange(refinement) / refinement
            z = np.r_[(knots[:-1, None] + np.diff(knots)[:, None] * fractions).ravel(), knots[-1]]
            start = time.perf_counter()
            table = BasisTable.build(
                ConcentricDisk(n_disk=96, n_limb=64),
                rho_grid=np.geomspace(1e-6, 10, n_rho),
                z_grid=z,
                astrometry=True,
            )
            duration = time.perf_counter() - start
            name = f"table_{n_rho}"
            available[name] = table
            path = args.output.with_name(args.output.stem + f"_{name}.npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            table.save(path)
            tables.append(
                dict(
                    method=name,
                    radii=n_rho,
                    ratios=len(z),
                    build_seconds=duration,
                    coefficient_bytes=int(table._coefficients.nbytes),
                    coordinate=table.coordinate,
                    representation=table.representation,
                    compressed_bytes=path.stat().st_size,
                    file=path.name,
                    generator="ConcentricDisk(n_disk=96,n_limb=64)",
                    timing_scope="Offline generation, not a controlled throughput timing",
                )
            )
            print(f"built {name}: {n_rho}x{len(z)} in {duration:.2f} s", flush=True)
    selected = (
        available if args.methods is None else {name: available[name] for name in args.methods}
    )
    return selected, tables


def accuracy(models, reference, quick):
    ratios = [0, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1, 1.001, 1.01, 2, 3, 10, 20, 30]
    if not quick:
        ratios = np.unique(
            np.r_[
                ratios,
                1 - np.geomspace(1e-6, 0.1, 11),
                1 + np.geomspace(1e-6, 0.1, 11),
                0.00499,
                0.005,
                0.00501,
                0.00999,
                0.01001,
                9.999,
                10.001,
                19.999,
                20.001,
            ]
        )
    evaluators = {name: jax.jit(model.joint) for name, model in models.items()}
    rows, checks = [], []
    for rho in RADII:
        positions = np.unique(np.r_[np.asarray(ratios) * rho, 0.1, 1.0, 10.0, 100.0])
        for a1 in COEFFICIENTS:
            truth = np.asarray([reference.joint(u, rho, a1) for u in positions])
            for name, evaluate in evaluators.items():
                magnification, shift = evaluate(jnp.asarray(positions), rho, a1=a1)
                for u, flux, offset, expected in zip(
                    positions, np.asarray(magnification), np.asarray(shift), truth
                ):
                    finite = bool(np.isfinite(flux) and np.isfinite(offset))
                    rows.append(
                        dict(
                            method=name,
                            rho=rho,
                            u=float(u),
                            ratio=float(u / rho),
                            a1=a1,
                            magnification=float(np.real(flux)),
                            reference_magnification=float(expected[0].real),
                            shift=float(np.real(offset)),
                            reference_shift=float(expected[1].real),
                            photometry_relative_error=float(np.real(flux) / expected[0].real - 1),
                            shift_absolute_error=float(abs(offset - expected[1])),
                            finite=finite,
                        )
                    )
        # Refine difficult contact cases and compare unrelated coordinate systems in their overlap.
        for z in (0.1, 0.999999, 1.0, 1.000001, 3, 10):
            u = z * rho
            _, _, s0, s1 = reference.basis(u, rho)
            refined = [
                independent.lens_centered_shift(u, rho, profile, a1, tolerance=2e-12)
                if z < 3
                else independent.wing_shift(u, rho, profile, a1, order=128)
                for profile, a1 in (("uniform", None), ("linear", 1.0))
            ]
            checks.append(
                dict(
                    rho=rho,
                    ratio=z,
                    max_absolute_change=float(np.max(np.abs(np.asarray(refined) - [s0, s1]))),
                )
            )
        reference.save()
        print(f"astrometry accuracy: rho={rho:g}, {len(positions)} positions", flush=True)
    return rows, checks


def radial_difference(reference, u, rho, a1, parameter, step):
    rows = []
    for offset in (-2, -1, 1, 2):
        p = dict(position=u, rho=rho, a1=a1)
        p[parameter] += step * offset
        f, s = reference.joint(**p)
        rows.append(np.array([f, s.real]))
    return (rows[0] - 8 * rows[1] + 8 * rows[2] - rows[3]) / (12 * step)


def wing_reference_derivatives(u, rho, a1, *, order):
    """Complex-step derivatives of an independent, nonsingular source integral.

    Finite differences cannot reliably recover the O(rho) radius derivative
    when the point-source term dominates. Analytic continuation of this smooth
    source-centered integral avoids that subtraction; no tested solver or JAX
    differentiation is used. Comparison of quadrature orders estimates its
    integration error independently of the complex-step size.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = (nodes + 1) * np.pi / 4
    radius, mu = np.sin(theta), np.cos(theta)
    weights = weights * radius * mu * (1 - a1 + a1 * mu)
    phi = np.arange(2 * order) * np.pi / order

    def integrate(position, source_radius):
        sx = source_radius * radius[:, None] * np.cos(phi)
        sy = source_radius * radius[:, None] * np.sin(phi)
        yx = position + sx
        distance = np.sqrt(yx * yx + sy * sy)
        h = np.sqrt(distance * distance + 4)
        excess = 4 / (distance * h * (distance * distance + 2 + distance * h))
        flux = np.sum(weights * (1 + excess).mean(axis=1)) / weights.sum()
        moment = np.sum(weights * (excess * sx + yx / (distance * h)).mean(axis=1)) / weights.sum()
        return np.array([flux, moment / flux])

    step = 1e-20 * max(u, rho)
    return np.column_stack(
        [
            np.imag(integrate(u + 1j * step, rho)) / step,
            np.imag(integrate(u, rho + 1j * step)) / step,
        ]
    )


def reference_jacobian(reference, u, rho, a1, angle):
    """Recover a Cartesian Jacobian from independent radial differences.

    At the origin, extrapolate the central slope from symmetric radial probes.
    This avoids differentiating a nondifferentiable polar coordinate in either
    the reference or the tested implementation.
    """
    flux, shift_complex = reference.joint(u, rho, a1)
    shift = shift_complex.real
    result = np.zeros((3, 4))
    uncertainty = np.zeros_like(result)
    wing_derivative = wing_uncertainty = None
    if u >= 3 * rho:
        wing_derivative = wing_reference_derivatives(u, rho, a1, order=96)
        wing_uncertainty = abs(wing_derivative - wing_reference_derivatives(u, rho, a1, order=64))
    if u == 0:
        coarse = reference.joint(1e-3 * rho, rho, a1)[1].real / (1e-3 * rho)
        fine = reference.joint(5e-4 * rho, rho, a1)[1].real / (5e-4 * rho)
        slope = fine + (fine - coarse) / 3
        result[1, 0] = result[2, 1] = slope
        uncertainty[1, 0] = uncertainty[2, 1] = abs(fine - coarse) / 3
        radial = np.array([0.0, slope])
    else:
        if wing_derivative is not None:
            radial, error = wing_derivative[:, 0], wing_uncertainty[:, 0]
        else:
            step = min(1e-4 * max(u, rho), abs(u - rho) / 32)
            coarse = radial_difference(reference, u, rho, a1, "position", step)
            fine = radial_difference(reference, u, rho, a1, "position", step / 2)
            radial = fine + (fine - coarse) / 15
            error = abs(fine - coarse) / 15
        c, s = np.cos(angle), np.sin(angle)
        result[0, :2] = radial[0] * np.array([c, s])
        coefficient = shift / u
        result[1:3, :2] = coefficient * np.eye(2) + (radial[1] - coefficient) * np.outer(
            [c, s], [c, s]
        )
        uncertainty[0, :2] = error[0]
        uncertainty[1:3, :2] = error[1]
    if wing_derivative is not None:
        derivative, error = wing_derivative[:, 1], wing_uncertainty[:, 1]
    else:
        step = min(1e-4 * rho, abs(u - rho) / 32)
        coarse = radial_difference(reference, u, rho, a1, "rho", step)
        fine = radial_difference(reference, u, rho, a1, "rho", step / 2)
        derivative = fine + (fine - coarse) / 15
        error = abs(fine - coarse) / 15
    direction = np.array([np.cos(angle), np.sin(angle)]) if u else np.zeros(2)
    result[:, 2] = np.r_[derivative[0], derivative[1] * direction]
    uncertainty[:, 2] = np.r_[error[0], error[1] * abs(direction)]
    f0, f1, s0, s1 = reference.basis(u, rho)
    d_gamma = 6 / (3 - a1) ** 2
    d_flux = d_gamma * (f1 - f0)
    d_shift = (d_gamma * (f1 * s1 - f0 * s0) - shift * d_flux) / flux
    result[:, 3] = np.r_[d_flux, d_shift * direction]
    return result, uncertainty


def gradients(models, reference, quick):
    def vector(model, p):
        x, y, rho, a1 = p
        flux, shift = model.joint(x + 1j * y, rho, a1=a1)
        return jnp.array([jnp.ravel(flux)[0], jnp.ravel(shift.real)[0], jnp.ravel(shift.imag)[0]])

    evaluators = {
        name: jax.jit(lambda p, m=model: (vector(m, p), jax.jacfwd(lambda q: vector(m, q))(p)))
        for name, model in models.items()
    }
    rows = []
    ratios = (
        [0, 0.5, 0.99, 1.01, 3, 20] if quick else [0, 0.001, 0.5, 0.99, 0.9999, 1.0001, 1.01, 3, 20]
    )
    for rho in RADII:
        positions = np.unique(np.r_[np.asarray(ratios) * rho, 1, 100])
        for u in positions:
            if u == rho:
                # Uniform brightness causes a singular contact derivative.
                continue
            for angle in [0, np.pi / 5] if u in (0, 0.5 * rho) else [np.pi / 5]:
                a1 = 0.5
                expected, uncertainty = reference_jacobian(reference, u, rho, a1, angle)
                p = jnp.asarray([u * np.cos(angle), u * np.sin(angle), rho, a1])
                reference_flux = reference.joint(u, rho, a1)[0]
                for name, evaluate in evaluators.items():
                    values, actual = evaluate(p)
                    actual, values = np.asarray(actual), np.asarray(values)
                    for output_index, output in enumerate(("A", "shift_x", "shift_y")):
                        for parameter_index, parameter in enumerate(("x", "y", "rho", "a1")):
                            actual_derivative = actual[output_index, parameter_index]
                            expected_derivative = expected[output_index, parameter_index]
                            scale = 1 if parameter == "a1" else rho
                            normalizer = reference_flux if output == "A" else 1.0
                            valid = bool(
                                np.isfinite(values[output_index]) and np.isfinite(actual_derivative)
                            )
                            error = abs(actual_derivative - expected_derivative)
                            rows.append(
                                dict(
                                    method=name,
                                    rho=rho,
                                    u=float(u),
                                    ratio=float(u / rho),
                                    angle=float(angle),
                                    a1=a1,
                                    output=output,
                                    parameter=parameter,
                                    value=float(actual_derivative),
                                    reference=float(expected_derivative),
                                    finite=valid,
                                    absolute_error=float(error) if valid else None,
                                    scaled_error=float(error * scale / normalizer)
                                    if valid
                                    else None,
                                    reference_scaled_uncertainty=float(
                                        uncertainty[output_index, parameter_index]
                                        * scale
                                        / normalizer
                                    ),
                                )
                            )
        reference.save()
        print(f"astrometry gradients: rho={rho:g}", flush=True)
    return rows


def runtimes(models, args):
    rows = []
    for name, model in models.items():
        for size in (1, 64, 512, 2048):
            if size == 2048 and name in ("disk_64_32", "disk_96_64"):
                rows.append(
                    dict(method=name, batch_size=size, skipped="Bound high-order AD memory use")
                )
                continue
            regimes = {
                "small_transit": (np.linspace(0.2, 1.8, size) * 1e-5, 1e-5),
                "large_transit": (np.linspace(0.2, 1.8, size), 1.0),
                "weak_wings": (np.geomspace(0.3, 10, size), 0.001),
            }

            def joint(p, r, a):
                return model.joint(p, r, a1=a)

            def shift(p, r, a):
                return model.shift(p, r, a1=a)

            def joint_objective(p, r, a):
                flux, offset = joint(p, r, a)
                return jnp.sum(jnp.log(flux)) + jnp.sum(offset.real**2 + offset.imag**2)

            def shift_objective(p, r, a):
                offset = shift(p, r, a)
                return jnp.sum(offset.real**2 + offset.imag**2)

            operations = {
                "joint": joint,
                "shift": shift,
                "joint_value_and_gradient": jax.value_and_grad(joint_objective, argnums=(0, 1, 2)),
                "shift_value_and_gradient": jax.value_and_grad(shift_objective, argnums=(0, 1, 2)),
            }
            first_u, first_r = regimes["small_transit"]
            initial = (jnp.asarray(first_u * np.exp(0.3j)), jnp.asarray(first_r), jnp.asarray(0.5))
            for operation, function in operations.items():
                start = time.perf_counter()
                compiled = jax.jit(function).lower(*initial).compile()
                compile_seconds = time.perf_counter() - start
                for regime, (u, rho) in regimes.items():
                    inputs = (jnp.asarray(u * np.exp(0.3j)), jnp.asarray(rho), jnp.asarray(0.5))
                    warm = synchronize(compiled(*inputs))
                    finite = all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(warm))
                    durations = []
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        synchronize(compiled(*inputs))
                        durations.append(time.perf_counter() - start)
                    rows.append(
                        dict(
                            method=name,
                            batch_size=size,
                            operation=operation,
                            regime=regime,
                            rho=rho,
                            finite=bool(finite),
                            compilation_seconds=compile_seconds,
                            warm_median_seconds=float(np.median(durations)),
                            warm_p10_seconds=float(np.quantile(durations, 0.1)),
                            warm_p90_seconds=float(np.quantile(durations, 0.9)),
                            repeats=args.repeats,
                        )
                    )
            print(f"astrometry runtime: {name}, N={size}", flush=True)
        jax.clear_caches()
    return rows


def likelihood_check(
    model, reference, times, physical, photo_data, photo_sigma, astro_data, covariance
):
    """Independent likelihood difference at one identical physical state."""
    position = times - physical["t0"] + 1j * physical["u0"]
    direct = [reference.joint(p, physical["rho"], physical["a1"]) for p in position]
    exact_flux = physical["source_flux"] * np.asarray([f for f, _ in direct])
    exact_shift = physical["theta_E"] * np.asarray([s for _, s in direct])
    baseline = np.array([physical["baseline_x"], physical["baseline_y"]])
    exact_position = np.column_stack([exact_shift.real, exact_shift.imag]) + baseline
    flux, shift = model.joint(jnp.asarray(position), physical["rho"], a1=physical["a1"])
    model_flux = physical["source_flux"] * np.asarray(flux)
    shift = physical["theta_E"] * np.asarray(shift)
    model_position = np.column_stack([shift.real, shift.imag]) + baseline
    precision = np.linalg.inv(covariance)

    def chi_squared(flux_values, positions):
        residual = astro_data - positions
        return float(
            np.sum(((photo_data - flux_values) / photo_sigma) ** 2)
            + np.einsum("ni,ij,nj->", residual, precision, residual)
        )

    exact_chi2 = chi_squared(exact_flux, exact_position)
    model_chi2 = chi_squared(model_flux, model_position)
    delta_photo = model_flux - exact_flux
    delta_position = model_position - exact_position
    bias_squared = float(
        np.sum((delta_photo / photo_sigma) ** 2)
        + np.einsum("ni,ij,nj->", delta_position, precision, delta_position)
    )
    return dict(
        delta_log_likelihood=-0.5 * (model_chi2 - exact_chi2),
        numerical_model_bias_mahalanobis_squared=bias_squared,
        exact_chi_squared=exact_chi2,
        model_chi_squared=model_chi2,
    )


def nuts(models, reference, args, checkpoint):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    results = []
    for scenario in args.scenarios:
        central = scenario == "transit"
        rho_true, u0_true = (0.1, 0.04) if central else (0.001, 0.3)
        theta_true, flux_true = 0.8, 1.0
        extent = 0.18 if central else 5.0
        t = np.linspace(-extent, extent, args.nuts_epochs)
        position = t + 1j * u0_true
        truth = [reference.joint(p, rho_true, 0.5) for p in position]
        true_flux = np.asarray([f for f, _ in truth]) * flux_true
        true_shift = np.asarray([s for _, s in truth]) * theta_true
        baseline = np.array([0.01, -0.02])
        astro_mean = np.column_stack([true_shift.real, true_shift.imag]) + baseline
        photo_sigma = 0.003 * true_flux
        astro_sigma = 0.001 * theta_true
        covariance = astro_sigma**2 * np.array([[1, 0.35], [0.35, 1.44]])
        rng = np.random.default_rng(args.seed + int(not central))
        observed_flux = true_flux + rng.normal(size=len(t)) * photo_sigma
        observed_position = astro_mean + rng.multivariate_normal([0, 0], covariance, size=len(t))
        reference.save()
        for name, evaluator in models.items():
            initial_flux, initial_shift = evaluator.joint(jnp.asarray(position), rho_true, a1=0.5)
            if not (
                np.isfinite(np.asarray(initial_flux)).all()
                and np.isfinite(np.asarray(initial_shift)).all()
            ):
                results.append(
                    dict(scenario=scenario, method=name, skipped="Truth outside solver domain")
                )
                checkpoint(results)
                continue
            print(f"joint NUTS: {scenario}, {name}, 2x({args.warmup}+{args.samples})", flush=True)

            def physical(q):
                return (
                    rho_true * jnp.exp(0.05 * q[..., 0]),
                    u0_true * jnp.exp(0.05 * q[..., 1]),
                    jax.nn.sigmoid(0.25 * q[..., 2]),
                    theta_true * jnp.exp(0.05 * q[..., 3]),
                    flux_true * jnp.exp(0.02 * q[..., 4]),
                    (0.001 if central else 0.02) * q[..., 5],
                    baseline[0] + astro_sigma * q[..., 6],
                    baseline[1] + astro_sigma * q[..., 7],
                )

            def model():
                q = numpyro.sample("q", dist.Normal(jnp.zeros(8), 2).to_event(1))
                rho, u0, a1, theta, flux_scale, t0, bx, by = physical(q)
                magnification, offset = evaluator.joint(jnp.asarray(t) - t0 + 1j * u0, rho, a1=a1)
                numpyro.sample(
                    "photometry",
                    dist.Normal(flux_scale * magnification, jnp.asarray(photo_sigma)),
                    obs=jnp.asarray(observed_flux),
                )
                predicted = theta * jnp.stack([offset.real, offset.imag], axis=-1) + jnp.array(
                    [bx, by]
                )
                numpyro.sample(
                    "astrometry",
                    dist.MultivariateNormal(predicted, covariance_matrix=jnp.asarray(covariance)),
                    obs=jnp.asarray(observed_position),
                )

            sampler = MCMC(
                NUTS(model, dense_mass=True, target_accept_prob=0.9, max_tree_depth=9),
                num_warmup=args.warmup,
                num_samples=args.samples,
                num_chains=2,
                chain_method="sequential",
                progress_bar=False,
            )
            key1, key2 = jax.random.split(jax.random.PRNGKey(args.seed))
            init = {"q": jnp.asarray(np.random.default_rng(args.seed + 1).normal(0, 0.1, (2, 8)))}
            start = time.perf_counter()
            sampler.warmup(key1, init_params=init)
            synchronize(sampler.post_warmup_state)
            warm_seconds = time.perf_counter() - start
            start = time.perf_counter()
            sampler.run(key2, extra_fields=("diverging", "num_steps", "accept_prob"))
            samples = np.asarray(sampler.get_samples(group_by_chain=True)["q"])
            sampling_seconds = time.perf_counter() - start
            keys = ("rho", "u0", "a1", "theta_E", "source_flux", "t0", "baseline_x", "baseline_y")
            posterior = {k: np.asarray(v) for k, v in zip(keys, physical(jnp.asarray(samples)))}
            diagnostics = summary(posterior, group_by_chain=True)
            metrics = {
                k: {field: float(v[field]) for field in ("mean", "std", "n_eff", "r_hat")}
                for k, v in diagnostics.items()
            }
            minimum = min(v["n_eff"] for v in metrics.values())
            rhat = max(v["r_hat"] for v in metrics.values())
            extras = sampler.get_extra_fields(group_by_chain=True)
            divergences = int(np.sum(extras["diverging"]))
            saturated = int(np.sum(np.asarray(extras["num_steps"]) >= 2**9 - 1))
            flux_error = np.asarray(initial_flux) * flux_true - true_flux
            shift_error = theta_true * np.asarray(initial_shift) - true_shift
            results.append(
                dict(
                    scenario=scenario,
                    method=name,
                    warmup_and_compile_seconds=warm_seconds,
                    sampling_seconds_including_sampling_compile=sampling_seconds,
                    total_seconds=warm_seconds + sampling_seconds,
                    min_ess=minimum,
                    max_rhat=rhat,
                    ess_per_sampling_second=minimum / sampling_seconds,
                    ess_per_total_second=minimum / (warm_seconds + sampling_seconds),
                    divergences=divergences,
                    tree_depth_saturated=saturated,
                    mean_leapfrog_steps=float(np.mean(extras["num_steps"])),
                    mean_acceptance=float(np.mean(extras["accept_prob"])),
                    posterior=metrics,
                    diagnostics_pass=bool(
                        divergences == 0 and saturated == 0 and rhat < 1.05 and minimum >= 100
                    ),
                    truth_max_photo_error_sigma=float(np.max(abs(flux_error) / photo_sigma)),
                    truth_max_astro_error_sigma=float(np.max(abs(shift_error) / astro_sigma)),
                    truth=dict(
                        rho=rho_true,
                        u0=u0_true,
                        a1=0.5,
                        theta_E=theta_true,
                        source_flux=flux_true,
                        t0=0,
                        baseline_x=baseline[0],
                        baseline_y=baseline[1],
                    ),
                )
            )
            validation_start = time.perf_counter()
            row = results[-1]
            state_mean = {key: value["mean"] for key, value in metrics.items()}
            row["likelihood_checks"] = {
                label: likelihood_check(
                    evaluator,
                    reference,
                    t,
                    physical_values,
                    observed_flux,
                    photo_sigma,
                    observed_position,
                    covariance,
                )
                for label, physical_values in (
                    ("truth", row["truth"]),
                    ("posterior_mean", state_mean),
                )
            }
            row["independent_validation_seconds_excluded_from_timing"] = (
                time.perf_counter() - validation_start
            )
            row["posterior_mean_minus_truth_in_sd"] = {
                key: (value["mean"] - row["truth"][key]) / value["std"]
                for key, value in metrics.items()
            }
            reference.save()
            if not row["diagnostics_pass"]:
                print(
                    f"WARNING: {scenario}/{name} failed chain diagnostics; "
                    "its ESS/s does not establish reliable posterior sampling.",
                    flush=True,
                )
            checkpoint(results)
            print(
                f"joint NUTS: {scenario}, {name}: minESS={minimum:.0f}, Rhat={rhat:.4f}, "
                f"ESS/s={minimum / sampling_seconds:.2f}",
                flush=True,
            )
            jax.clear_caches()
    for scenario in args.scenarios:
        baseline = next(
            (
                row
                for row in results
                if row["scenario"] == scenario
                and row["method"] == "disk_64_32"
                and "posterior" in row
            ),
            None,
        )
        if baseline is None:
            continue
        for row in results:
            if row["scenario"] != scenario or "posterior" not in row:
                continue
            row["posterior_comparison_to_disk_64_32"] = {
                key: dict(
                    mean_difference_in_reference_sd=(
                        value["mean"] - baseline["posterior"][key]["mean"]
                    )
                    / baseline["posterior"][key]["std"],
                    mean_difference_in_combined_mcse=(
                        value["mean"] - baseline["posterior"][key]["mean"]
                    )
                    / np.sqrt(
                        value["std"] ** 2 / value["n_eff"]
                        + baseline["posterior"][key]["std"] ** 2
                        / baseline["posterior"][key]["n_eff"]
                    ),
                )
                for key, value in row["posterior"].items()
            }
            row["posterior_comparison_diagnostics_pass"] = bool(
                row["diagnostics_pass"] and baseline["diagnostics_pass"]
            )
    return dict(
        results=results,
        timing_scope=NUTS_TIMING_SCOPE,
        posterior_comparison_scope=NUTS_COMPARISON_SCOPE,
        truth_astro_error_scope=NUTS_ASTRO_ERROR_SCOPE,
        chains=2,
        warmup_per_chain=args.warmup,
        samples_per_chain=args.samples,
        seed=args.seed,
        epochs=args.nuts_epochs,
        astrometric_correlation=0.35 / 1.2,
        priors="q~Normal(0,2); rho/u0/theta_E exponential(.05q), flux exponential(.02q), "
        "a1=sigmoid(.25q), t0=.001q transit/.02q wings, baseline=truth+.001theta_E*q",
        scope="Two controlled synthetic scenarios and one noise realization each; fixed Einstein "
        "crossing time and orientation, no blend or proper motion. Same data and priors "
        "for every method. Report diagnostics and approximation errors with ESS/s.",
    )


def summarize(report):
    result = []
    for name in report["methods"]:
        for rho in RADII:
            all_values = [
                r for r in report.get("accuracy", []) if r["method"] == name and r["rho"] == rho
            ]
            valid = [r for r in all_values if r["finite"]]
            derivatives = [
                r
                for r in report.get("gradients", [])
                if r["method"] == name and r["rho"] == rho and r["output"] != "A" and r["finite"]
            ]
            result.append(
                dict(
                    method=name,
                    rho=rho,
                    samples=len(all_values),
                    valid_samples=len(valid),
                    max_shift_absolute_error=max(
                        (r["shift_absolute_error"] for r in valid), default=None
                    ),
                    max_photometry_relative_error=max(
                        (abs(r["photometry_relative_error"]) for r in valid), default=None
                    ),
                    max_shift_derivative_scaled_error=max(
                        (r["scaled_error"] for r in derivatives), default=None
                    ),
                    max_shift_derivative_absolute_error=max(
                        (r["absolute_error"] for r in derivatives), default=None
                    ),
                )
            )
    return result


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")


def plot(report, output):
    """Compare one representative configuration per numerical method."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    representatives = (
        ("fft_4096", "FFTLog", "#247BA0"),
        ("disk_64_32", "Direct quadrature", "#E07A5F"),
        ("table_192", "Basis table", "#3D8067"),
    )
    for name, label, color in representatives:
        if name not in report["methods"]:
            continue
        rows = sorted((r for r in report["summary"] if r["method"] == name), key=lambda r: r["rho"])
        for axis, field in zip(
            axes[:2], ("max_shift_absolute_error", "max_shift_derivative_scaled_error")
        ):
            axis.loglog(
                [r["rho"] for r in rows],
                [max(r[field], 1e-18) if r[field] is not None else np.nan for r in rows],
                "o-",
                label=label,
                color=color,
            )
        rows = sorted(
            (
                r
                for r in report.get("runtime", [])
                if r["method"] == name
                and not r.get("skipped")
                and r["regime"] == "large_transit"
                and r["operation"] == "joint_value_and_gradient"
            ),
            key=lambda r: r["batch_size"],
        )
        axes[2].loglog(
            [r["batch_size"] for r in rows],
            [1e3 * r["warm_median_seconds"] for r in rows],
            "o-",
            color=color,
        )
    axes[0].set(xlabel="Source radius rho", ylabel="Maximum absolute shift error / theta_E")
    axes[1].set(
        xlabel="Source radius rho", ylabel="Maximum scaled shift derivative error / theta_E"
    )
    axes[2].set(xlabel="Epochs (rho=1, shared radius)", ylabel="Warm joint value + gradient (ms)")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="lower center", fontsize=10)
    fig.suptitle("Astrometric accuracy and runtime")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("accuracy", "gradients", "runtime"),
        default=["accuracy", "gradients", "runtime"],
    )
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--with-tables", action="store_true")
    parser.add_argument("--table-sizes", nargs="+", type=int, choices=(96, 192))
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--nuts", action="store_true")
    parser.add_argument("--nuts-only", action="store_true")
    parser.add_argument(
        "--nuts-methods",
        nargs="+",
        default=["fft_4096", "fft_8192", "disk_32_16", "disk_64_32", "table_96", "table_192"],
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=("transit", "weak_wings"),
        default=["transit", "weak_wings"],
    )
    parser.add_argument("--warmup", type=int, default=350)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--nuts-epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=261023)
    parser.add_argument("--output", type=Path, default=OUTPUT / "astrometry_methods.json")
    parser.add_argument(
        "--resume", action="store_true", help="Retain prior sections in the output report"
    )
    args = parser.parse_args()
    if args.samples < 100 or args.warmup < 100 or args.repeats < 3:
        parser.error("Use >=100 warmup/samples and >=3 timing repeats.")
    if args.nuts_only:
        args.nuts = True
    if args.nuts:
        args.with_tables = True
    jax.config.update("jax_enable_x64", True)
    reference = Reference()
    selected, tables = make_methods(args)
    report = dict(
        **metadata(),
        methods=list(selected),
        tables=tables,
        reference_signature=reference.signature,
        benchmark_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        derivative_reference="Independent radial finite differences and central-slope extrapolation; "
        "source-centered complex-step integration in weak wings, refined from order64 to96",
        quick=args.quick,
        scope="Finite sampled domain; no uniform bounds. Absolute shifts are in theta_E. "
        "Derivative scaled_error multiplies x/y/rho errors by rho and a1 errors by 1; "
        "photometric errors also divide by A. Exact contact derivatives excluded. "
        "Every timed call synchronizes; compilation separated, identical executable "
        "reused across small-transit, large-transit, and weak-wing regimes.",
    )
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text())
        if previous["methods"] != report["methods"]:
            parser.error("Resumed report must use the same selected methods.")
        previous.update(report)
        report = previous
    report.setdefault("section_source_sha256", {})
    report.setdefault("section_benchmark_source_sha256", {})
    if not args.nuts_only:
        if "accuracy" in args.sections:
            report["accuracy"], report["reference_checks"] = accuracy(
                selected, reference, args.quick
            )
            report["section_source_sha256"]["accuracy"] = report["source_sha256"]
            report["section_benchmark_source_sha256"]["accuracy"] = report[
                "benchmark_source_sha256"
            ]
            save(report, args.output)
        if "gradients" in args.sections:
            report["gradients"] = gradients(selected, reference, args.quick)
            report["section_source_sha256"]["gradients"] = report["source_sha256"]
            report["section_benchmark_source_sha256"]["gradients"] = report[
                "benchmark_source_sha256"
            ]
            save(report, args.output)
        if "runtime" in args.sections:
            report["runtime"] = runtimes(selected, args)
            report["section_source_sha256"]["runtime"] = report["source_sha256"]
            report["section_benchmark_source_sha256"]["runtime"] = report["benchmark_source_sha256"]
            save(report, args.output)
        report["summary"] = summarize(report)
        save(report, args.output)
        if all(key in report for key in ("accuracy", "gradients", "runtime")):
            plot(report, args.output)
    if args.nuts:

        def checkpoint(results):
            report["nuts"] = dict(results=results, incomplete=True)
            save(report, args.output)

        report["nuts"] = nuts(
            {name: selected[name] for name in args.nuts_methods}, reference, args, checkpoint
        )
        report["section_source_sha256"]["nuts"] = report["source_sha256"]
        report["section_benchmark_source_sha256"]["nuts"] = report["benchmark_source_sha256"]
    save(report, args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
