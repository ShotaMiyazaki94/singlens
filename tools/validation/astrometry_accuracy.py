"""Independent finite-source astrometry checks across source sizes and separations.

Run ``python -m tools.validation.astrometry_accuracy``. The references integrate
the point-lens image moment directly; they use no singlens FFT or magnification.
Errors are in Einstein-radius units because the shift crosses zero at the limb.
"""

import hashlib
import inspect
import json
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import quad_vec

import singlens as sl

OUTPUT = Path(__file__).parent / "outputs"
RADII = np.array([1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0])
RATIOS = np.unique(
    np.r_[
        0,
        0.001,
        0.00999,
        0.01,
        0.01001,
        0.1,
        0.5,
        0.99,
        1,
        1.01,
        2,
        5,
        9.99,
        10,
        10.01,
        19.99,
        20,
        20.01,
        100,
        np.linspace(0.99, 1.01, 41),
    ]
)
PROFILES = (("uniform", None), ("linear", 1.0), ("mu2", None))


def _intensity(mu2, source, a1):
    mu2 = np.maximum(mu2, 0.0)
    if source == "uniform":
        return np.ones_like(mu2)
    if source == "linear":
        return 1 - a1 + a1 * np.sqrt(mu2)
    if source == "mu2":
        return mu2
    raise ValueError(f"Unknown source profile: {source}")


@lru_cache(None)
def lens_centered_shift(u, rho, source="uniform", a1=None, *, tolerance=2e-10):
    """Adaptive image-moment integral, regular even with the lens in the source.

    In lens-centered polar coordinates the area Jacobian cancels the 1/r
    point-magnification singularity. A sine coordinate along each source chord
    also removes the square-root limb-intensity endpoint. Distances along the
    chord are scaled by rho, retaining accuracy for very small sources.
    """
    if u == 0:
        return 0.0
    if rho == 0:
        return u / (u * u + 2)
    if source == "linear" and a1 is None:
        raise ValueError("The linear reference requires a1.")
    z = u / rho

    def angular(theta):
        center = z * np.cos(theta)
        half_chord = np.sqrt(max(0.0, 1 - (z * np.sin(theta)) ** 2))
        if half_chord == 0:
            return np.zeros(2)
        lower = max(0.0, center - half_chord)
        start = np.arcsin(np.clip((lower - center) / half_chord, -1, 1))

        def radial(psi):
            chord_mu = half_chord * np.cos(psi)
            x = center + half_chord * np.sin(psi)
            p = rho * x
            h = np.hypot(p, 2)
            intensity = _intensity(chord_mu * chord_mu, source, a1)
            jacobian = chord_mu
            return intensity * jacobian / h * np.array([p * p + 2, x * (p * p + 3) * np.cos(theta)])

        return quad_vec(radial, start, np.pi / 2, epsabs=tolerance * 0.02, epsrel=tolerance)[0]

    upper = np.pi if z < 1 else np.pi / 2 if z == 1 else np.arcsin(1 / z)
    denominator, moment = quad_vec(angular, 0, upper, epsabs=tolerance * 0.02, epsrel=tolerance)[0]
    return float(rho * moment / denominator - u)


@lru_cache(None)
def wing_shift(u, rho, source="uniform", a1=None, *, order=64):
    """Smooth source-centered quadrature for u >= 3*rho.

    Accumulate the shift numerator directly rather than subtracting u from a
    centroid of size u. With image-plane centroid c(y), A(c_x-u) is written
    (A-1)*s_x + y_x/(|y|*sqrt(|y|**2+4)); the unweighted s_x integral is zero.
    """
    if rho == 0:
        return u / (u * u + 2)
    if u < 3 * rho:
        raise ValueError("Use lens-centered integration for u < 3*rho.")
    x, weights = np.polynomial.legendre.leggauss(order)
    theta = (x + 1) * np.pi / 4
    radius, mu = np.sin(theta), np.cos(theta)
    weights = weights * radius * mu * _intensity(mu * mu, source, a1)
    phi = np.arange(2 * order) * np.pi / order
    sx = rho * radius[:, None] * np.cos(phi)
    sy = rho * radius[:, None] * np.sin(phi)
    yx = u + sx
    distance = np.hypot(yx, sy)
    h = np.hypot(distance, 2)
    excess = 4 / (distance * h * (distance * distance + 2 + distance * h))
    denominator = np.sum(weights * (1 + excess).mean(axis=1))
    numerator = np.sum(weights * (excess * sx + yx / (distance * h)).mean(axis=1))
    return float(numerator / denominator)


def reference_shift(u, rho, source="uniform", a1=None):
    """Signed shift along the positive source axis, in Einstein-radius units."""
    if rho == 0:
        return u / (u * u + 2)
    if u >= 3 * rho:
        return wing_shift(u, rho, source, a1)
    return lens_centered_shift(u, rho, source, a1)


def reference_signature():
    code = "".join(
        inspect.getsource(f) for f in (_intensity, lens_centered_shift, wing_shift, reference_shift)
    )
    return hashlib.sha256(code.encode()).hexdigest()


def main():
    jax.config.update("jax_enable_x64", True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUTPUT / "astrometry_reference.json"
    signature = reference_signature()
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache = cached.get("values", {}) if cached.get("signature") == signature else {}
    rows, max_wing_change, max_overlap_change = [], 0.0, 0.0
    for source, a1 in PROFILES:
        model = sl.Model(source=source)
        evaluate = jax.jit(model.astrometric_shift_at)
        for rho in RADII:
            # Absolute u samples also exercise the weak-lensing wings when rho is tiny.
            positions = np.unique(np.r_[RATIOS * rho, np.maximum([1, 3, 10, 30, 100], 3 * rho)])
            reference = []
            for u in positions:
                key = f"{source}:{a1}:{rho:.16g}:{u:.16g}"
                if key not in cache:
                    cache[key] = reference_shift(float(u), float(rho), source, a1)
                reference.append(cache[key])
            cache_path.write_text(json.dumps(dict(signature=signature, values=cache)))
            for u in np.maximum([1, 3, 10, 100], 3 * rho):
                coarse = wing_shift(float(u), float(rho), source, a1)
                refined = wing_shift(float(u), float(rho), source, a1, order=96)
                max_wing_change = max(max_wing_change, abs(coarse - refined))
            overlap = lens_centered_shift(3 * float(rho), float(rho), source, a1)
            max_overlap_change = max(
                max_overlap_change,
                abs(overlap - wing_shift(3 * float(rho), float(rho), source, a1)),
            )
            actual = np.asarray(evaluate(jnp.asarray(positions), rho=rho, a1=a1)).real
            errors = actual - np.asarray(reference)
            # Regression guard for the measured implementation, not a mathematical
            # error bound. Absolute errors remain meaningful at shift zero crossings.
            target = 1.5e-3 * min(float(rho), 1.0)
            worst = int(np.argmax(abs(errors)))
            row = dict(
                source=source,
                a1=a1,
                rho=float(rho),
                target_absolute_error=target,
                max_absolute_error=float(np.max(abs(errors))),
                worst_u_over_rho=float(positions[worst] / rho),
                separations=positions.tolist(),
                shift=actual.tolist(),
                reference=reference,
                absolute_error=errors.tolist(),
            )
            rows.append(row)
            print(
                f"{source}, rho={rho:g}: max |error|={row['max_absolute_error']:.6g} theta_E"
                f" at u/rho={row['worst_u_over_rho']:.6g}",
                flush=True,
            )
    report = dict(
        package_version=sl.__version__,
        jax_version=jax.__version__,
        fft=asdict(sl.FFTConfig()),
        reference="Independent image-moment adaptive integration and source-centered wing quadrature",
        reference_signature=signature,
        radii=RADII.tolist(),
        ratios=RATIOS.tolist(),
        error_unit="theta_E",
        wing_reference_max_absolute_change=max_wing_change,
        reference_overlap_max_absolute_difference=max_overlap_change,
        samples=sum(len(row["reference"]) for row in rows),
        results=rows,
        scope="Sampled values, not a uniform error bound. Uniform, a1=1 linear limb darkening, "
        "and the pure mu**2 profile are tested. Source center, limb contact, branch "
        "boundaries, and weak-lensing wings are included. Derivatives are tested separately.",
    )
    path = OUTPUT / "astrometry_accuracy.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if (
        max_wing_change > 1e-11
        or max_overlap_change > 1e-9
        or any(
            not np.isfinite(row["shift"]).all()
            or row["max_absolute_error"] >= row["target_absolute_error"]
            for row in rows
        )
    ):
        raise RuntimeError(
            f"Astrometric accuracy target or reference convergence failed; see {path}"
        )
    print(f"{report['samples']} values pass; report: {path}")


if __name__ == "__main__":
    main()
