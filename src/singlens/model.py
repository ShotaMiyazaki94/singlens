"""Prepared single-lens models, separated from differentiable parameters."""

from dataclasses import dataclass, field, fields

import jax.numpy as jnp

from .lensing import FFTConfig, LinearLimbDarkened, QuadraticProfile, UniformDisk
from .lensing.point import pspl
from .parameters import Parameters
from .photometry import fit_flux
from .trajectory import AnnualParallax


@dataclass(frozen=True, kw_only=True, eq=False)
class Model:
    """A single-lens model with a chosen source profile and sky geometry.

    ``source`` is ``'point'`` (default), ``'uniform'``, ``'linear'``, or ``'mu2'``.
    The latter is the pure ``I(mu)=mu**2`` radial profile. Finite-source models
    require ``Parameters.rho``; linear models also require ``Parameters.a1``.
    Physical coefficients and timescales belong in ``Parameters``.

    Compile a bound method directly: ``jax.jit(model.magnification)``. All
    evaluation methods preserve the time/position shape. ``jax.grad`` and
    ``jax.jacfwd`` accept a whole ``Parameters`` object; vectorize independent
    events with ``jax.vmap(..., in_axes=(None, 0))``.
    """

    source: str = "point"
    parallax: AnnualParallax | None = None
    fft: FFTConfig | None = None
    _solver: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.source not in ("point", "uniform", "linear", "mu2"):
            raise ValueError("source must be 'point', 'uniform', 'linear', or 'mu2'.")
        if self.parallax is not None and not isinstance(self.parallax, AnnualParallax):
            raise TypeError("parallax must be an AnnualParallax instance.")
        if self.fft is not None and not isinstance(self.fft, FFTConfig):
            raise TypeError("fft must be an FFTConfig instance.")
        if self.source == "point":
            if self.fft is not None:
                raise ValueError("FFT settings apply only to finite-source models.")
            return
        config = self.fft if self.fft is not None else FFTConfig()
        object.__setattr__(self, "fft", config)
        factory = {"uniform": UniformDisk, "linear": LinearLimbDarkened, "mu2": QuadraticProfile}[
            self.source
        ]
        object.__setattr__(self, "_solver", factory(fft=config))

    def _validate(self, params):
        if not isinstance(params, Parameters):
            raise TypeError("params must be a Parameters instance.")
        for item in fields(params):
            value = getattr(params, item.name)
            if value is not None and jnp.ndim(value) != 0:
                raise ValueError(
                    f"{item.name} must be scalar per event; use jax.vmap to batch events."
                )
        self._validate_source(params.rho, params.a1)
        if self.parallax is None:
            if params.pi_e_n is not None or params.pi_e_e is not None:
                raise ValueError("Parallax amplitudes require Model(parallax=AnnualParallax(...)).")
        elif params.pi_e_n is None or params.pi_e_e is None:
            raise ValueError("A parallax model requires both pi_e_n and pi_e_e (zero is allowed).")

    def _validate_source(self, rho, a1):
        if self.source == "point":
            if rho is not None or a1 is not None:
                raise ValueError(
                    "Point-source models do not take rho or a1; select a finite source."
                )
        elif rho is None:
            raise ValueError("Finite-source models require rho.")
        if self.source == "linear":
            if a1 is None:
                raise ValueError("Linear limb darkening requires a1.")
        elif a1 is not None:
            raise ValueError("a1 applies only to source='linear'.")
        for name, value in (("rho", rho), ("a1", a1)):
            if value is not None and jnp.ndim(value) != 0:
                raise ValueError(f"{name} must be scalar; use jax.vmap for multiple values.")
            if value is not None and jnp.iscomplexobj(value):
                raise ValueError(f"{name} must be real.")

    def trajectory(self, times, params):
        """Complex source position ``tau + 1j*beta`` at the supplied times."""
        self._validate(params)
        times = jnp.asarray(times)
        safe_t_e = jnp.where(jnp.isfinite(params.t_e) & (params.t_e > 0), params.t_e, jnp.nan)
        tau = (times - params.t0) / safe_t_e
        beta = jnp.broadcast_to(params.u0, times.shape)
        if self.parallax is not None:
            dtau, dbeta = self.parallax.offsets(times, pi_e_n=params.pi_e_n, pi_e_e=params.pi_e_e)
            tau, beta = tau + dtau, beta + dbeta
        return tau + 1j * beta

    def magnification_at(self, position, *, rho=None, a1=None):
        """Evaluate real separation or complex position directly, without a trajectory.

        Invalid radii and linear coefficients outside [0, 1] return NaN under
        both eager execution and JIT. ``rho=0`` is the point-source limit.
        """
        self._validate_source(rho, a1)
        position = jnp.asarray(position)
        if self.source == "point":
            return pspl(position)
        u = jnp.abs(position)
        if self.source == "linear":
            a1 = jnp.asarray(a1)
            valid = jnp.isfinite(a1) & (a1 >= 0) & (a1 <= 1)
            safe_a1 = jnp.where(valid, a1, 0.0)
            result = jnp.where(valid, self._solver.A(u, rho, a1=safe_a1), jnp.nan)
        else:
            result = self._solver.A(u, rho)
        return result.reshape(position.shape)

    def magnification(self, times, params):
        """Evaluate a magnification light curve, preserving ``times.shape``."""
        return self.magnification_at(self.trajectory(times, params), rho=params.rho, a1=params.a1)

    def flux(self, times, params, *, f_s=1.0, f_b=0.0):
        """Observed flux with scalar coefficients: ``f_s*magnification + f_b``."""
        if jnp.ndim(f_s) != 0 or jnp.ndim(f_b) != 0:
            raise ValueError("f_s and f_b must be scalar for one dataset; use jax.vmap to batch.")
        return f_s * self.magnification(times, params) + f_b

    def fit_flux(self, times, params, observed_flux, flux_err):
        """Fit source/blend fluxes for one dataset; return a named ``FluxFit``.

        This profiles over the two flux coefficients at fixed event parameters.
        It is not Bayesian marginalization. For HMC, sample fluxes with explicit
        priors, or use a Gaussian-prior marginal likelihood from
        ``singlens.photometry``. Reported flux errors are conditional on params.
        """
        return fit_flux(self.magnification(times, params), observed_flux, flux_err)
