"""Circular finite-source astrometry from flux and first-moment convolutions.

The dark point lens is at the origin. All lengths, including the returned
source-relative displacement, are in angular Einstein-radius units.
"""

import jax.numpy as jnp
from jax import lax

from ..numerics.fftlog import hankel
from ..numerics.interpolate import cubic_interp
from .finite import (
    LinearLimbDarkened,
    QuadraticProfile,
    _damping,
    _disk_transform,
    _linear_transform,
    _theta,
    _weights,
)
from .point import point_source_shift


def _blend(x):
    x = jnp.clip(x, 0.0, 1.0)
    return x**3 * (10 + x * (-15 + 6 * x))


class FiniteSourceAstrometry:
    """Prepared moment kernels for an existing circular source profile.

    Construct outside transformations (or under ensure_compile_time_eval),
    then compile ``shift``. The profile's magnification kernels are unchanged.
    """

    def __init__(self, profile):
        self.profile = profile
        self.fft = profile.fft
        r = profile._grid
        h = jnp.hypot(r, 2.0)
        p = r * r + 2 + r * h
        # q = A_point * centroid_point - r has a 1/r tail. Transform
        # T = div(q) instead: T=2(A_point-1)-4/(r*h**3) decays as 8/r**6.
        # These rationalized inputs avoid cancellation in both far tails.
        a_input = 4 * r / (h * p)
        t_input = 16 * r * (1 + 1 / p) / (h**3 * p)
        self.k, a = hankel(r, a_input, nu=1.25).hankel(0)
        _, t = hankel(r, t_input, nu=1.25).hankel(0)
        z2 = (self.k * r[0]) ** 2
        tail1 = r[0] * (1 - z2 / 12 + z2**2 / 320 - z2**3 / 16128)
        tail2 = r[0] ** 2 / 2 * (1 - z2 / 8 + z2**2 / 192 - z2**3 / 9216)
        self._ak = 2 * jnp.pi * (a + tail1 - tail2)
        self._tk = 2 * jnp.pi * (t + 1.5 * tail1 - 2 * tail2)

        def small_transform(sk):
            return hankel(r, sk * _damping(r), nu=1.0, N_pad=self.fft.n_fft).hankel(1)

        if isinstance(profile, LinearLimbDarkened):
            self._small_u, self._small_disk = small_transform(_disk_transform(r))
            _, self._small_limb = small_transform(_linear_transform(r))
        else:
            self._small_u, self._small_v = small_transform(profile.sk(r, 1.0))

    def _central_coefficient(self, rho, z2, **profile):
        # Return (centroid-source)/u through O((u/rho)**2), including
        # the exact central slope. The numerator is M/u-A, formed directly.
        h = jnp.hypot(rho, 2.0)
        disk = (-1 / h + z2 * (4 / h**3 - 6 / h**5)) / rho

        def moment(n):
            mu = jnp.cos(jnp.asarray(_theta))
            x = rho * jnp.sin(jnp.asarray(_theta))
            inv_h = 1 / jnp.hypot(x, 2.0)
            weights = jnp.asarray(_weights)
            constant = -2 * jnp.sum(weights * mu ** (n + 1) * inv_h**3)
            curvature = n * jnp.sum(weights * mu ** (n - 1) * (2 * inv_h**3 - 3 * inv_h**5))
            return (n + 2) / rho * (constant + z2 * curvature)

        if isinstance(self.profile, LinearLimbDarkened):
            a1 = profile.get("a1", self.profile.a1)
            gamma = 2 * a1 / (3 - a1)
            numerator = (1 - gamma) * disk + gamma * moment(1)
        elif isinstance(self.profile, QuadraticProfile):
            numerator = moment(2)
        else:
            numerator = disk
        return numerator / self.profile._central_expansion(rho, z2, **profile)

    def _small_coefficient(self, u, rho, **profile):
        z = u / rho
        if isinstance(self.profile, LinearLimbDarkened):
            a1 = profile.get("a1", self.profile.a1)
            gamma = 2 * a1 / (3 - a1)
            v_grid = (1 - gamma) * self._small_disk + gamma * self._small_limb
        else:
            v_grid = self._small_v
        v = cubic_interp(z, self._small_u, v_grid)
        # K_point(r)=3/2*r_hat+O(r**2); the flux is H(z)/rho.
        return 1.5 * v / (z * self.profile.Aext0(z, **profile)) - 1

    def _full_coefficient(self, u, rho, **profile):
        sk = self.profile.sk(self.k, rho, **profile) * _damping(self.k * rho)
        padding = max(512, self.fft.n_fft // 8)
        ua, a = hankel(self.k, self.k**2 * self._ak * sk, nu=1.5, N_pad=padding).hankel(0)
        uq, q = hankel(self.k, self.k * self._tk * sk, nu=1.0, N_pad=padding).hankel(1)
        # J0 and J1 have different reciprocal output grids. Interpolate each
        # on its own grid, retaining the signed shift at source contact.
        log_u = jnp.log10(u)
        excess = cubic_interp(log_u, jnp.log10(ua), a / (2 * jnp.pi))
        q = cubic_interp(log_u, jnp.log10(uq), q / (2 * jnp.pi))
        return (q / u - excess) / (1 + excess)

    def _far_fraction(self, u, rho, **profile):
        # Source-average expansion through <r**2>; inverse lengths retain
        # the 1/u astrometric tail even when photometric excess is negligible.
        inv_u = 1 / u
        inv_h = 1 / jnp.hypot(u, 2.0)
        inv_d = 1 / jnp.hypot(u, jnp.sqrt(2.0))
        moment = self.profile._radial_moment(**profile)
        return 4 * moment * (rho * inv_u) ** 2 * (2 + 3 * inv_u**2) * (u * inv_d) ** 2 * inv_h**2

    def shift(self, position, rho, **profile):
        """Return signed real/complex displacement, preserving position shape.

        ``rho`` is scalar; use vmap for multiple source radii. Invalid radii
        return NaN. Exact alignment has zero displacement and a finite slope.
        """
        position = jnp.asarray(position)
        position = position.astype(jnp.result_type(position, 1.0))
        rho = jnp.asarray(rho)
        if rho.ndim != 0:
            raise ValueError("rho must be scalar; use jax.vmap for multiple radii.")
        if jnp.iscomplexobj(rho):
            raise ValueError("rho must be real.")
        valid = jnp.isfinite(rho) & (rho >= 0)
        safe_rho = jnp.where(valid & (rho > 0), rho, 1.0)
        # The radius is not differentiable at the origin. Mask it before
        # sqrt, and use the even central series there to keep Cartesian AD.
        radius2 = jnp.real(position) ** 2 + jnp.imag(position) ** 2
        center = radius2 < (0.005 * safe_rho) ** 2
        u = jnp.hypot(
            jnp.where(center, 0.005 * safe_rho, jnp.real(position)),
            jnp.where(center, 0.0, jnp.imag(position)),
        )
        outer = self.fft.u_switch * safe_rho
        far_u = jnp.maximum(u, outer / 2)
        far_fraction = self._far_fraction(far_u, safe_rho, **profile)
        far = (1 - far_fraction) / jnp.hypot(far_u, jnp.sqrt(2.0)) ** 2

        def extended(_):
            return lax.cond(
                safe_rho < self.fft.rho_switch,
                lambda r: self._small_coefficient(u, r, **profile),
                lambda r: self._full_coefficient(u, r, **profile),
                safe_rho,
            )

        coefficient = lax.cond(jnp.all(u >= outer) | (rho == 0), lambda _: far, extended, None)
        weight = _blend(2 * u / outer - 1)
        coefficient = jnp.where(weight >= 1, far, coefficient + weight * (far - coefficient))
        z2 = jnp.where(radius2 < (0.01 * safe_rho) ** 2, radius2 / safe_rho**2, 0.0)
        central = self._central_coefficient(safe_rho, z2, **profile)
        center_weight = _blend((u / safe_rho - 0.005) / 0.005)
        coefficient = jnp.where(
            center_weight >= 1,
            coefficient,
            central + center_weight * (coefficient - central),
        )
        point = point_source_shift(position)
        result = jnp.where(u >= outer, point * (1 - far_fraction), position * coefficient)
        result = jnp.where(rho == 0, point, result)
        return jnp.where(valid, result, jnp.nan)
