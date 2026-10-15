"""FFT finite-source magnification, adapted from microJAX and Sugiyama (2022).

The FFT and small-source branches are numerical approximations. Natural cubic
interpolation is C2 inside each grid, but branch thresholds can still be
nonsmooth. See docs/numerics.md.
"""

import jax.numpy as jnp
import numpy as np
from jax import lax

from ..numerics.fftlog import hankel
from ..numerics.interpolate import cubic_interp
from ..numerics.special import j1, j2
from .config import FFTConfig
from .point import pspl

# Central magnifications: r=sin(theta) removes the limb endpoint singularity.
_nodes, _weights = np.polynomial.legendre.leggauss(64)
_theta = (_nodes + 1) * np.pi / 4
_weights = _weights * np.pi / 4


def _damping(x):
    # Paper's 50*k_rho cutoff, with its O(k**2) low-frequency bias removed.
    # The steeper tail suppresses undersampled modes and stabilizes derivatives.
    gaussian = jnp.exp(-((x / (100 * np.pi)) ** 2))
    return gaussian * (2 - gaussian)


def _central_disk(rho, z2):
    h = jnp.hypot(rho, 2.0)
    return (h - 4 * z2 / h**3) / rho


def _central_moment(rho, n, z2):
    r, mu = jnp.sin(jnp.asarray(_theta)), jnp.cos(jnp.asarray(_theta))
    x = rho * r
    h = jnp.hypot(x, 2.0)
    center = jnp.sum(jnp.asarray(_weights) * mu ** (n + 1) * (x * x + 2) / h)
    # A radial profile is a weighted mixture of uniform disks. Integrating
    # their central curvatures gives the u**2 term without differentiating
    # the singular point-lens integrand.
    curvature = jnp.sum(jnp.asarray(_weights) * mu ** (n - 1) / h**3)
    return (n + 2) / rho * (center - 2 * n * z2 * curvature)


def _disk_transform(x):
    safe = jnp.where(jnp.abs(x) < 1e-3, 1.0, x)
    series = 1 - x * x / 8 + x**4 / 192 - x**6 / 9216
    return jnp.where(jnp.abs(x) < 1e-3, series, 2 * j1(safe) / safe)


def _linear_transform(x):
    safe = jnp.where(jnp.abs(x) < 1e-2, 1.0, x)
    series = 1 - x * x / 10 + x**4 / 280 - x**6 / 15120
    regular = 3 * (jnp.sin(safe) - safe * jnp.cos(safe)) / safe**3
    return jnp.where(jnp.abs(x) < 1e-2, series, regular)


class _FiniteSource:
    """Base for normalized, circular source profiles.

    Construct once outside JIT, then compile the bound ``A`` method. ``rho``
    must be scalar; use ``jax.vmap`` for multiple source radii. ``A`` retains
    microJAX's at-least-1D output convention. No global JAX options are changed.
    """

    def __init__(self, *, fft=None):
        self.fft = FFTConfig() if fft is None else fft
        if not isinstance(self.fft, FFTConfig):
            raise TypeError("fft must be an FFTConfig instance.")
        self._grid = jnp.logspace(self.fft.log_u_min, self.fft.log_u_max, self.fft.n_fft)
        self.init_Apk()
        self.init_Aext0()

    def init_Apk(self):
        u = self._grid
        # Rationalized A_p - 1 avoids cancellation in the far wings.
        h = jnp.hypot(u, 2.0)
        u2Au = 4 * u / (h * (u * u + 2 + u * h))
        self.k, apk = hankel(u, u2Au, nu=1.5).hankel(0)
        # Restore the missing 0<u<u_min integral of u*(A_p-1)*J0(k*u).
        # A_p=1/u+O(u), and k*u_min<=1/2 on this reciprocal grid.
        z2 = (self.k * u[0]) ** 2
        tail = u[0] * (1 - z2 / 12 + z2**2 / 320 - z2**3 / 16128)
        tail -= u[0] ** 2 / 2 * (1 - z2 / 8 + z2**2 / 192 - z2**3 / 9216)
        self.apk = (apk + tail) * 2 * jnp.pi

    def init_Aext0(self):
        x = self._grid
        fx = x * self.sk(x, 1.0) * _damping(x)
        self._small_u, self._small_a = hankel(x, fx, nu=1.5, N_pad=self.fft.n_fft).hankel(0)

    def Aext0(self, x):
        return cubic_interp(x, self._small_u, self._small_a)

    def sk(self, k, rho):
        """Normalized source Fourier transform (subclass hook)."""
        raise NotImplementedError

    def A0(self, rho, **profile):
        """Central magnification."""
        return self._central_expansion(rho, 0.0, **profile)

    def _central_expansion(self, rho, z2, **profile):
        """Central value plus curvature, with z2=(u/rho)**2 (subclass hook)."""
        raise NotImplementedError

    def _radial_moment(self, **profile):
        """Mean squared source radius in units of rho**2."""
        return 0.5

    def _far_magnification(self, u, rho, **profile):
        # Source average: A + <r**2>/4 * laplacian(A). Inverse lengths
        # avoid powers overflowing at large u. The next term is O((rho/u)**4).
        inv_u, inv_h = 1 / u, 1 / jnp.hypot(u, 2.0)
        correction = (
            8
            * self._radial_moment(**profile)
            * (rho * inv_u) ** 2
            * (1 - 3 * inv_h**2)
            * inv_u
            * inv_h**3
        )
        return pspl(u) + correction

    def _A_for_small_rho(self, u, rho, **profile):
        near = u < self.fft.u_switch * rho
        # Safe inactive branches are essential for reverse-mode autodiff.
        x = jnp.where(near, u / rho, 1.0)
        # A_p(u)=1/u+O(u); its source average has no additive constant.
        extended = self.Aext0(x, **profile) / rho
        point = pspl(jnp.where(near, 1.0, u))
        return jnp.where(near, extended, point)

    def _large_rho_grid(self, rho, **profile):
        """Convolved magnification grid; independent of observation epochs."""
        damping = _damping(self.k * rho)
        cj = self.apk * self.k**2 * self.sk(self.k, rho, **profile) * damping
        # Preserve the validated default; finer grids need a wider logarithmic
        # guard interval to prevent wrap-around near the central expansion.
        padding = 512 if self.fft.n_fft <= 8192 else self.fft.n_fft // 8
        u_fft, a_fft = hankel(self.k, cj, nu=1.5, N_pad=padding).hankel(0)
        a_fft = a_fft / (2 * jnp.pi) + 1.0
        return jnp.log10(u_fft), a_fft

    def _A_for_large_rho(self, u, rho, **profile):
        log_grid, a_fft = self._large_rho_grid(rho, **profile)
        # Use a strictly increasing interpolation grid, including in the wings.
        log_u = jnp.log10(jnp.where(u > 0, u, 1.0))
        values = cubic_interp(log_u, log_grid, a_fft, right=1.0)
        return jnp.where(u >= 100, 1.0, values)

    def A(self, u, rho):
        """Evaluate at separations ``u`` and scalar radius ``rho``.

        ``rho=0`` uses the point-source limit; negative/nonfinite radii return
        NaN, including under JIT. Exact alignment uses a stable central value.
        """
        return self._evaluate(u, rho)

    def _evaluate(self, u, rho, **profile):
        u = jnp.atleast_1d(jnp.abs(jnp.asarray(u)))
        rho = jnp.asarray(rho)
        if rho.ndim != 0:
            raise ValueError("rho must be scalar; use jax.vmap for multiple radii.")
        valid = jnp.isfinite(rho) & (rho >= 0)
        safe_rho = jnp.where(valid & (rho > 0), rho, 1.0)
        near_center = u < 0.01 * safe_rho
        safe_u = jnp.where(near_center, 0.01 * safe_rho, u)
        outer = self.fft.u_switch * safe_rho
        far = self._far_magnification(jnp.maximum(u, outer / 2), safe_rho, **profile)

        def finite_source(_):
            return lax.cond(
                safe_rho < self.fft.rho_switch,
                lambda r: self._A_for_small_rho(safe_u, r, **profile),
                lambda r: self._A_for_large_rho(safe_u, r, **profile),
                safe_rho,
            )

        # One event shares one FFT. Omit it only when every position is far.
        values = lax.cond(jnp.all(u >= outer) | (rho == 0), lambda _: far, finite_source, None)
        # The same per-position expression in mixed and all-far batches avoids
        # batch-dependent answers. Smoothly join over u/rho=10..20 by default.
        x = jnp.clip(2 * u / outer - 1, 0.0, 1.0)
        weight = x**3 * (10 + x * (-15 + 6 * x))
        values = jnp.where(x >= 1, far, values + weight * (far - values))
        # FFT endpoints are unreliable arbitrarily close to alignment. The
        # central expansion retains the correct value and leading derivative.
        z2 = jnp.where(near_center, u / safe_rho, 0.0) ** 2
        central = self._central_expansion(safe_rho, z2, **profile)
        values = jnp.where(near_center, central, values)
        point = pspl(jnp.where(rho == 0, u, 1.0))
        values = jnp.where(rho == 0, point, values)
        return jnp.where(valid, values, jnp.nan)


class UniformDisk(_FiniteSource):
    """Uniform-brightness circular source."""

    def sk(self, k, rho):
        return _disk_transform(jnp.asarray(k) * rho)

    def _central_expansion(self, rho, z2):
        return _central_disk(rho, z2)


class LinearLimbDarkened(_FiniteSource):
    r"""Linear limb darkening: ``I(mu) = 1 - a1*(1-mu)``.

    Here ``mu = sqrt(1 - (r/rho)**2)``. The disk-averaged intensity is
    ``1-a1/3``; internally the normalized mixture coefficient is
    ``Gamma = 2*a1/(3-a1)``. This corrects microJAX's coefficient mismatch.
    ``A(u, rho, a1=...)`` accepts a dynamic coefficient. The two small-source
    basis kernels are prepared once; the full branch uses a single FFT.
    """

    def __init__(self, a1=0.5, *, fft=None):
        self.a1 = a1
        super().__init__(fft=fft)

    @property
    def gamma(self):
        return 2 * self.a1 / (3 - self.a1)

    def sk(self, k, rho, a1=None):
        gamma = self.gamma if a1 is None else 2 * a1 / (3 - a1)
        x = jnp.asarray(k) * rho
        return (1 - gamma) * _disk_transform(x) + gamma * _linear_transform(x)

    def _central_expansion(self, rho, z2, a1=None):
        gamma = self.gamma if a1 is None else 2 * a1 / (3 - a1)
        return (1 - gamma) * _central_disk(rho, z2) + gamma * _central_moment(rho, 1, z2)

    def _radial_moment(self, a1=None):
        gamma = self.gamma if a1 is None else 2 * a1 / (3 - a1)
        return 0.5 - 0.1 * gamma

    def init_Aext0(self):
        x = self._grid
        weight = x * _damping(x)
        self._small_u, self._small_disk = hankel(
            x, weight * _disk_transform(x), nu=1.5, N_pad=self.fft.n_fft
        ).hankel(0)
        _, self._small_limb = hankel(
            x, weight * _linear_transform(x), nu=1.5, N_pad=self.fft.n_fft
        ).hankel(0)

    def Aext0(self, x, a1=None):
        gamma = self.gamma if a1 is None else 2 * a1 / (3 - a1)
        values = (1 - gamma) * self._small_disk + gamma * self._small_limb
        return cubic_interp(x, self._small_u, values)

    def A(self, u, rho, *, a1=None):
        """Evaluate with the constructor's coefficient or a dynamic ``a1``."""
        return self._evaluate(u, rho, a1=self.a1 if a1 is None else a1)


class QuadraticProfile(_FiniteSource):
    """Pure radial profile ``I(mu) = mu**2`` (not a two-coefficient LD law)."""

    def sk(self, k, rho):
        x = jnp.asarray(k) * rho
        safe = jnp.where(jnp.abs(x) < 0.1, 1.0, x)
        series = 1 - x * x / 12 + x**4 / 384 - x**6 / 23040 + x**8 / 2211840
        return jnp.where(jnp.abs(x) < 0.1, series, 8 * j2(safe) / safe**2)

    def _central_expansion(self, rho, z2):
        return _central_moment(rho, 2, z2)

    def _radial_moment(self):
        return 1 / 3
