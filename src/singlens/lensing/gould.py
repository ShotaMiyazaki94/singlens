"""Gould--Yoo approximation and controlled higher-order small-source moments."""

import jax.numpy as jnp

from .concentric import ConcentricDisk, _disk_average


class GouldYoo(ConcentricDisk):
    r"""Differentiable small-source linear limb darkening, evaluated by quadrature.

    ``order=0`` is the conventional Gould--Yoo approximation
    ``A_PS(u) * [B0(u/rho) - Gamma*B1(u/rho)]``. Its basis functions are
    evaluated with fixed quadrature, rather than a bundled lookup table.

    ``order=2`` and ``order=4`` instead integrate the point-source series
    ``1/s + 3*s/8`` and ``1/s + 3*s/8 - 5*s**3/128`` over the brightness
    profile. These are moment corrections, not the conventional approximation
    multiplied by a correction factor. Their leading relative truncation
    errors at fixed ``u/rho`` scale as rho**4 and rho**6. We require the
    interior convergence domain ``u+rho<2``; outside it they return NaN. Even within it, accuracy
    degrades near the convergence boundary. Quadrature error is additional.

    These methods are intended for explicit small-source experiments. They
    do not switch silently to FFTLog or exact integration at large radius.
    Increasing quadrature order does not remove small-source truncation error.
    """

    def __init__(self, a1=0.5, *, n_disk=64, n_limb=32, split=True, order=0):
        if order not in (0, 2, 4) or isinstance(order, bool):
            raise ValueError("order must be 0, 2, or 4.")
        self.order = order
        super().__init__(a1=a1, n_disk=n_disk, n_limb=n_limb, split=split)

    def _disk(self, u, rho):
        return _disk_average(u, rho, self._nodes, self._weights, order=self.order)

    def joint_basis(self, u, rho):
        raise NotImplementedError(
            "GouldYoo implements photometry only; use ConcentricDisk for astrometry."
        )

    def joint(self, position, rho, *, a1=None):
        raise NotImplementedError(
            "GouldYoo implements photometry only; use ConcentricDisk for astrometry."
        )

    def shift(self, position, rho, *, a1=None):
        raise NotImplementedError(
            "GouldYoo implements photometry only; use ConcentricDisk for astrometry."
        )

    def basis(self, u, rho):
        disk, limb = super().basis(u, rho)
        u = jnp.atleast_1d(jnp.abs(jnp.asarray(u)))
        if self.order == 0:
            # u*A_PS(u), with its exact finite value at u=0. Evaluate with
            # inverse hypot to avoid squaring very large u.
            h = jnp.hypot(u, 2.0)
            factor = h - 2 / h
            factor = jnp.where(jnp.asarray(rho) == 0, 1.0, factor)
            return factor * disk, factor * limb
        valid = (u + jnp.asarray(rho) < 2) | (jnp.asarray(rho) == 0)
        return jnp.where(valid, disk, jnp.nan), jnp.where(valid, limb, jnp.nan)
