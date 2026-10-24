"""Check FFT magnification accuracy across source sizes and limb contact.

Run with ``python -m tools.validation.accuracy_domain``. Independent integral
references are cached in the development output directory, outside the package.
"""

import hashlib
import inspect
import json
from dataclasses import asdict
from importlib import import_module
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad

from example._common import metadata, save_json
from singlens.lensing import FFTConfig, LinearLimbDarkened, QuadraticProfile, UniformDisk

accuracy = import_module("example.compare-single-vbml.code.accuracy")
OUTPUT = Path(__file__).parent / "outputs"
RADII = np.unique(np.r_[np.geomspace(1e-6, 10, 29), 9.99e-5, 1.001e-4])
RATIOS = np.unique(
    np.r_[
        0,
        np.geomspace(1e-8, 0.5, 20),
        np.geomspace(0.01, 0.5, 80),
        [0.00999, 0.01000001, 0.01001, 0.0101],
        np.linspace(0.5, 0.95, 10),
        np.linspace(0.95, 1.05, 101),
        np.linspace(0.98, 1.02, 161),
        np.linspace(1.1, 2, 10),
        [3, 5, 9.99, 10, 10.01, 15, 19.99, 20, 20.01, 30, 50],
    ]
)


def quadratic_integral(u, rho, *, tolerance=1e-11):
    """Independent lens-centered integral for the normalized mu**2 profile."""
    z = u / rho

    def angular(theta):
        delta = np.sqrt(max(0.0, 1 - (z * np.sin(theta)) ** 2))
        center = z * np.cos(theta)
        lo, hi = max(0.0, center - delta), center + delta

        def radial(x):
            mu2 = max(0.0, 1 - (x - center) ** 2 - (z * np.sin(theta)) ** 2)
            return mu2 * ((rho * x) ** 2 + 2) / np.hypot(rho * x, 2)

        return quad(radial, lo, hi, epsabs=tolerance * 0.01, epsrel=tolerance)[0]

    upper = np.pi if z < 1 else np.pi / 2 if z == 1 else np.arcsin(1 / z)
    value = quad(angular, 0, upper, epsabs=tolerance * 0.01, epsrel=tolerance, limit=200)[0]
    return 4 * value / (np.pi * rho)


def reference_signature():
    code = inspect.getsource(accuracy.integral) + inspect.getsource(quadratic_integral)
    return hashlib.sha256(code.encode()).hexdigest()


def wing_integral(u, rho, power, *, order=64):
    """Independent nonsingular source-centered quadrature for u >= 3*rho."""
    x, weights = np.polynomial.legendre.leggauss(order)
    theta = (x + 1) * np.pi / 4
    radius, mu = np.sin(theta), np.cos(theta)
    weights = weights * radius * mu ** (power + 1)
    phi = np.arange(2 * order) * np.pi / order
    distance = np.sqrt(
        u**2 + (rho * radius[:, None]) ** 2 - 2 * u * rho * radius[:, None] * np.cos(phi)
    )
    values = (distance**2 + 2) / (distance * np.hypot(distance, 2))
    return np.sum(weights * values.mean(axis=1)) / weights.sum()


def main():
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT / "accuracy_domain_reference.json"
    signature = reference_signature()
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache = cached.get("values", {}) if cached.get("signature") == signature else {}
    rows, max_wing_change = [], 0.0
    targets = {"uniform": 1.1e-3, "linear_a1_1": 1e-3, "quadratic": 1e-3}
    for source, power, model in (
        ("uniform", 0, UniformDisk()),
        ("linear_a1_1", 1, LinearLimbDarkened(1.0)),
        ("quadratic", 2, QuadraticProfile()),
    ):
        evaluate = jax.jit(model.A)
        for rho in RADII:
            reference = []
            for ratio in RATIOS:
                key = f"{source}:{rho:.16g}:{ratio:.16g}"
                if key not in cache:
                    cache[key] = (
                        quadratic_integral(ratio * rho, rho)
                        if power == 2
                        else accuracy.integral(ratio * rho, rho, a1=float(power), tolerance=1e-11)
                    )
                reference.append(cache[key])
            cache_path.write_text(json.dumps(dict(signature=signature, values=cache)))
            wing_u = np.maximum([0.03, 0.1, 0.3, 1, 3, 10, 30, 99], 3 * rho)
            wing = np.array([wing_integral(u, rho, power) for u in wing_u])
            refined = np.array([wing_integral(u, rho, power, order=96) for u in wing_u])
            max_wing_change = max(max_wing_change, float(np.max(abs(wing / refined - 1))))
            positions = np.r_[RATIOS * rho, wing_u]
            reference = np.r_[reference, refined]
            actual = np.asarray(evaluate(jnp.asarray(positions), rho))
            errors = actual / reference - 1
            worst = int(np.argmax(abs(errors)))
            rows.append(
                dict(
                    source=source,
                    rho=float(rho),
                    max_relative_error=float(np.max(abs(errors))),
                    rms_relative_error=float(np.sqrt(np.mean(errors**2))),
                    worst_u_over_rho=float(positions[worst] / rho),
                    separations=positions.tolist(),
                    magnification=actual.tolist(),
                    reference=reference.tolist(),
                    relative_error=errors.tolist(),
                )
            )
            print(f"{source}, rho={rho:g}: {100 * rows[-1]['max_relative_error']:.6f}%", flush=True)
    report = dict(
        **metadata(),
        fft=asdict(FFTConfig()),
        radii=RADII.tolist(),
        u_over_rho=RATIOS.tolist(),
        target_relative_error_by_profile=targets,
        reference="Independent lens-centered adaptive quadrature and source-centered wing quadrature",
        reference_signature=signature,
        wing_reference_max_relative_change=max_wing_change,
        samples=sum(len(row["reference"]) for row in rows),
        max_relative_error=max(row["max_relative_error"] for row in rows),
        scope="Sampled values, not a uniform error bound. Uniform and a1=1 bracket all physical linear coefficients 0<=a1<=1 by positivity of their normalized mixture. The pure mu**2 profile is checked separately. Includes near alignment, limb contact, approximation boundaries, and far wings.",
        results=rows,
    )
    save_json(report, "accuracy_domain", OUTPUT)
    if max_wing_change > 1e-10 or any(
        row["max_relative_error"] >= targets[row["source"]] for row in rows
    ):
        raise RuntimeError("FFT accuracy target or reference convergence failed; see report.")
    print(
        f"{report['samples']} values pass; max relative error {report['max_relative_error']:.6g}."
    )


if __name__ == "__main__":
    main()
