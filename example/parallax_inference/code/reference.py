"""Independent outside-source integration using smooth exact coordinates."""

from functools import cache

import numpy as np


@cache
def outside_rule(order):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    chi = (nodes + 1) * np.pi / 4
    phi = (nodes + 1) * np.pi / 2
    return (
        np.sin(chi),
        np.cos(chi),
        weights * np.pi / 4,
        np.cos(phi),
        np.sin(phi) ** 2 * weights * np.pi / 2,
    )


def outside_integral(u, rho, *, a1, order=64, chunk=64):
    """Integrate the exact point-lens kernel for ``u/rho >= 1.05``.

    With z=u/rho, substitute theta=asin(sin(chi)/z), chi in [0, pi/2].
    Each ray's source segment is x=center+delta*cos(phi), phi in [0, pi],
    where center=sqrt(z²-sin²chi) and delta=coschi. The angular Jacobian
    is delta/center, while mu*dx=delta²sin²phi*dphi. These substitutions
    remove both endpoint square roots. The uniform radial term uses its
    analytic primitive. No point-source or small-radius approximation enters.
    """
    u = np.asarray(u, dtype=float)
    flat = u.ravel()
    if not np.isfinite(rho) or rho <= 0 or not 0 <= a1 <= 1:
        raise ValueError("Positive radius and 0<=a1<=1 required")
    if np.any(~np.isfinite(flat)) or np.any(flat / rho < 1.05):
        raise ValueError("Outside quadrature requires finite u/rho>=1.05")
    sin_chi, delta, chi_weights, cos_phi, phi_weights = outside_rule(order)
    result = np.empty_like(flat)
    for start in range(0, len(flat), chunk):
        ratio = flat[start : start + chunk, None] / rho
        center = np.sqrt(ratio**2 - sin_chi**2)
        lo, hi = center - delta, center + delta
        p, q = rho * lo, rho * hi
        hp, hq = np.hypot(p, 2), np.hypot(q, 2)
        uniform = delta * (hq + p * (q + p) / (hq + hp))
        separation = rho * (center[..., None] + delta[:, None] * cos_phi)
        kernel = (separation**2 + 2) / np.hypot(separation, 2)
        limb = delta**2 * np.sum(kernel * phi_weights, axis=-1)
        angular = ((1 - a1) * uniform + a1 * limb) * delta / center
        result[start : start + chunk] = np.sum(angular * chi_weights, axis=-1)
    return (2 * result / (np.pi * rho * (1 - a1 / 3))).reshape(u.shape)
