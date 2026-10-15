"""JAX-native FFTLog and Hankel utilities used by the FSPL solver.

Adapted from microJAX and fft-extended-source (see NOTICE). This keeps
the code JIT-friendly and differentiable. It provides a minimal FFTLog class
for logarithmically spaced inputs and a thin ``hankel`` wrapper that evaluates
zero-th order Hankel transforms (spherical Bessel ``j_ell`` integrals) that
appear in the FFT-based microlensing algorithm.
"""

from functools import partial

import jax.numpy as jnp
from jax import jit
from jax.numpy.fft import irfft, rfft
from jax.scipy.special import loggamma


class fftlog:
    """FFTLog on a log-spaced grid with optional edge extrapolation and padding."""

    def __init__(
        self, x, fx, nu=1.1, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0
    ):
        """Prepare FFT coefficients for a biased input function ``f(x)/x^nu``.

        Parameters
        ----------
        x : array_like
            Monotonic, logarithmically spaced sample positions.
        fx : array_like
            Function values ``f(x)`` on the same grid.
        nu : float, optional
            Bias exponent applied before the FFT (default 1.1).
        N_extrap_low, N_extrap_high : int, optional
            Number of points to extrapolate on each side in log space to reduce
            ringing near the boundaries.
        c_window_width : float, optional
            Fraction of the upper half of coefficients to cosine-taper.
        N_pad : int, optional
            Zero-padding length (added symmetrically) to improve FFT stability.
        """
        self.dlnx = jnp.log(x[1] / x[0])
        self.nu = nu
        self.N_extrap_low = N_extrap_low
        self.N_extrap_high = N_extrap_high
        self.c_window_width = c_window_width

        # extrapolate x and f(x) linearly in log(x), and log(f(x))
        self.x = log_extrap(x, N_extrap_low, N_extrap_high)
        self.fx = log_extrap(fx, N_extrap_low, N_extrap_high)
        self.N = self.x.size

        # zero-padding
        if N_pad:
            pad = jnp.zeros(N_pad)
            self.x = log_extrap(self.x, N_pad, N_pad)
            self.fx = jnp.hstack((pad, self.fx, pad))
            self.N += 2 * N_pad
            self.N_extrap_high += N_pad
            self.N_extrap_low += N_pad

        if self.N % 2 == 1:  # Make sure the array sizes are even
            self.x = self.x[:-1]
            self.fx = self.fx[:-1]
            self.N -= 1
            if N_pad:
                self.N_extrap_high -= 1

        self.m, self.c_m = self.get_c_m()
        self.eta_m = 2 * jnp.pi / (float(self.N) * self.dlnx) * self.m

    def get_c_m(self):
        """Compute smoothed FFT coefficients of the biased function ``f/x^nu``."""
        f_b = self.fx * self.x ** (-self.nu)
        c_m = rfft(f_b)
        m = jnp.arange(0, self.N // 2 + 1)
        c_m = c_m * c_window(m, int(self.c_window_width * self.N // 2.0))
        return m, c_m

    def fftlog(self, ell):
        r"""
        Calculate ``F(y) = ∫ dx/x f(x) j_ell(xy)`` on the reflected grid ``y``.

        Parameters
        ----------
        ell : int or float
            Order of the spherical Bessel ``j_ell``.

        Returns
        -------
        y : jnp.ndarray
            Output sample locations ordered like ``x[::-1]`` without padding.
        Fy : jnp.ndarray
            Transform values on ``y``.
        """
        z_ar = self.nu + 1j * self.eta_m
        y = (ell + 1.0) / self.x[::-1]
        h_m = self.c_m * (self.x[0] * y[0]) ** (-1j * self.eta_m) * g_l(ell, z_ar)

        Fy = irfft(jnp.conj(h_m)) * y ** (-self.nu) * jnp.sqrt(jnp.pi) / 4.0
        return y[self.N_extrap_high : self.N - self.N_extrap_low], Fy[
            self.N_extrap_high : self.N - self.N_extrap_low
        ]


class hankel:
    """Zero-th order Hankel transform wrapper built on :class:`fftlog`."""

    def __init__(self, x, fx, nu, N_extrap_low=0, N_extrap_high=0, c_window_width=0.25, N_pad=0):
        self.myfftlog = fftlog(
            x, jnp.sqrt(x) * fx, nu, N_extrap_low, N_extrap_high, c_window_width, N_pad
        )

    def hankel(self, n):
        """Evaluate ``∫ dx/x fx(x) J_n(xy)`` via the FFTLog core."""
        y, Fy = self.myfftlog.fftlog(n - 0.5)
        Fy *= jnp.sqrt(2 * y / jnp.pi)
        return y, Fy


### Utility functions ####################
@partial(jit, static_argnums=(1, 2))
def log_extrap(x, N_extrap_low, N_extrap_high):
    """Extrapolate a log-spaced grid ``x`` by a fixed number of steps on each side."""
    if N_extrap_low > 0:
        dlnx_low = jnp.log(x[1] / x[0])
        low_x = x[0] * jnp.exp(dlnx_low * jnp.arange(-N_extrap_low, 0))
    else:
        low_x = jnp.array([])

    if N_extrap_high > 0:
        dlnx_high = jnp.log(x[-1] / x[-2])
        high_x = x[-1] * jnp.exp(dlnx_high * jnp.arange(1, N_extrap_high + 1))
    else:
        high_x = jnp.array([])

    x_extrap = jnp.concatenate((low_x, x, high_x))
    return x_extrap


@partial(jit, static_argnums=(1,))
def c_window(n, n_cut):
    """Cosine taper for the highest ``n_cut`` frequency bins to suppress ringing."""
    if n_cut <= 1:
        return jnp.ones_like(n, dtype=float)
    n_right = n[-1] - n_cut
    idx = n > n_right
    theta_right = (n[-1] - n) / (n[-1] - n_right - 1).astype(float)
    W = jnp.where(
        idx, theta_right - 1 / (2 * jnp.pi) * jnp.sin(2 * jnp.pi * theta_right), jnp.ones_like(n)
    )
    return W


@jit
def g_m_vals(mu, q):
    """Gamma ratio evaluated in log space, including large imaginary arguments."""
    q = jnp.asarray(q, dtype=complex)
    numerator = (mu + 1 + q) / 2
    denominator = (mu + 1 - q) / 2
    # Reciprocal Gamma vanishes at denominator poles. Keep the inactive
    # evaluation finite so autodiff does not inherit a NaN from loggamma(0).
    pole = (
        (jnp.imag(denominator) == 0)
        & (jnp.real(denominator) <= 0)
        & (jnp.real(denominator) == jnp.floor(jnp.real(denominator)))
    )
    ratio = jnp.exp(loggamma(numerator) - loggamma(jnp.where(pole, 1.0, denominator)))
    return jnp.where(pole, 0.0, ratio)


def g_l(ell, z_array):
    """Gamma prefactor for spherical Bessel integrals."""
    return 2.0**z_array * g_m_vals(ell + 0.5, z_array - 1.5)
