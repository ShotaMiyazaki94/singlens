"""Prepared single-lens models, separated from differentiable parameters."""

from dataclasses import dataclass, field, fields
from functools import cached_property

import jax
import jax.numpy as jnp

from .lensing import (
    BasisTable,
    FFTConfig,
    LinearLimbDarkened,
    QuadraticProfile,
    UniformDisk,
)
from .lensing.basis_table import _bundled_basis_table
from .lensing.point import point_source_shift, pspl
from .parameters import Parameters
from .photometry import fit_flux, marginalize_flux
from .trajectory import AnnualParallax


@dataclass(frozen=True, kw_only=True, eq=False)
class Model:
    """A single-lens model with a chosen source profile and sky geometry.

    Choose ``source='point'``, ``'uniform'``, ``'linear'``, or ``'mu2'`` explicitly.
    The latter is the pure ``I(mu)=mu**2`` radial profile. Finite-source models
    require ``Parameters.rho``; linear models also require ``Parameters.a1``.
    Physical coefficients and timescales belong in ``Parameters``.

    Uniform and linear sources use a bundled photometric/astrometric table
    for ``1e-6 <= rho <= 10`` and ``abs(position)/rho <= 1e8``. Outside this
    domain results are NaN; ``rho=0`` gives the analytic point-source limit.
    Pass ``solver='fft'`` for FFTLog with fixed settings. ``solver='table'``
    explicitly selects the bundled table. The ``mu2`` profile uses FFTLog.
    Selection is fixed at construction, with no parameter-dependent switching.

    Enable JAX float64 before constructing models or JAX arrays.
    Compile a bound method directly: ``jax.jit(model.magnification)``. All
    evaluation methods preserve the time/position shape. ``jax.grad`` and
    ``jax.jacfwd`` accept a whole ``Parameters`` object; vectorize independent
    events with ``jax.vmap(..., in_axes=(None, 0))``.
    """

    source: str
    parallax: AnnualParallax | None = None
    solver: str | FFTConfig | BasisTable | None = None
    _solver: object = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.source not in ("point", "uniform", "linear", "mu2"):
            raise ValueError("source must be 'point', 'uniform', 'linear', or 'mu2'.")
        if self.parallax is not None and not isinstance(self.parallax, AnnualParallax):
            raise TypeError("parallax must be an AnnualParallax instance.")
        if isinstance(self.solver, str):
            if self.solver not in ("fft", "table"):
                raise ValueError("solver must be 'fft' or 'table'.")
        elif self.solver is not None and not isinstance(self.solver, (FFTConfig, BasisTable)):
            raise TypeError("solver must be 'fft', 'table', or an expert solver configuration.")
        if self.source == "point":
            if self.solver is not None:
                raise ValueError("Solver settings apply only to finite-source models.")
            return
        solver = self.solver
        if isinstance(solver, str):
            if solver == "fft":
                solver = FFTConfig()
            elif self.source in ("uniform", "linear"):
                solver = _bundled_basis_table(jax.config.x64_enabled)
            else:
                raise ValueError("The table solver supports source='uniform' or 'linear'.")
        if solver is None:
            solver = (
                FFTConfig()
                if self.source == "mu2"
                else _bundled_basis_table(jax.config.x64_enabled)
            )
        if isinstance(solver, BasisTable):
            if self.source not in ("uniform", "linear"):
                raise ValueError("A BasisTable supports source='uniform' or 'linear'.")
            object.__setattr__(self, "solver", solver)
            object.__setattr__(self, "_solver", solver)
            return
        object.__setattr__(self, "solver", solver)
        factory = {"uniform": UniformDisk, "linear": LinearLimbDarkened, "mu2": QuadraticProfile}[
            self.source
        ]
        object.__setattr__(self, "_solver", factory(fft=solver))

    @cached_property
    def _astrometric_solver(self):
        # Photometry-only models do not prepare the extra moment kernels.
        # The first astrometric call may occur inside jit/grad: cached arrays
        # must be concrete rather than tracers belonging to that first call.
        if isinstance(self._solver, BasisTable):
            return self._solver
        from .lensing.astrometry import FiniteSourceAstrometry

        with jax.ensure_compile_time_eval():
            return FiniteSourceAstrometry(self._solver)

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
                raise ValueError("Parallax amplitudes require an AnnualParallax geometry on Model.")
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
        elif isinstance(self._solver, BasisTable):
            result = self._solver.A(u, rho, a1=0.0)
        else:
            result = self._solver.A(u, rho)
        return result.reshape(position.shape)

    def magnification(self, times, params):
        """Evaluate a magnification light curve, preserving ``times.shape``."""
        return self.magnification_at(self.trajectory(times, params), rho=params.rho, a1=params.a1)

    def astrometric_shift_at(self, position, *, rho=None, a1=None):
        """Centroid shift from the unlensed source center, in ``theta_E`` units.

        The unresolved images are produced by a dark point lens at the origin.
        Real positions give signed real shifts and complex positions give
        complex shifts in the same coordinate frame. Input shape is preserved.
        ``rho=0`` gives the point-source limit. Invalid radii or linear
        coefficients outside [0, 1] return NaN, including under JIT.

        This is a dimensionless lensing shift. An angular Einstein radius,
        sky rotation, source proper motion/parallax, and the flux and positions
        of blended light are needed to model an observed sky centroid.
        """
        self._validate_source(rho, a1)
        position = jnp.asarray(position)
        if self.source == "point":
            return point_source_shift(position)
        if self.source == "linear":
            a1 = jnp.asarray(a1)
            valid = jnp.isfinite(a1) & (a1 >= 0) & (a1 <= 1)
            safe_a1 = jnp.where(valid, a1, 0.0)
            result = jnp.where(
                valid, self._astrometric_solver.shift(position, rho, a1=safe_a1), jnp.nan
            )
        elif isinstance(self._solver, BasisTable):
            result = self._astrometric_solver.shift(position, rho, a1=0.0)
        else:
            result = self._astrometric_solver.shift(position, rho)
        return result.reshape(position.shape)

    def joint_at(self, position, *, rho=None, a1=None):
        """Return ``(magnification, astrometric_shift)`` at the same positions.

        Table solvers share their flux/moment calculation; FFTLog
        evaluates the two quantities separately. Results match the separate
        methods and have the shape of ``position``.
        """
        self._validate_source(rho, a1)
        if not isinstance(self._solver, BasisTable):
            return (
                self.magnification_at(position, rho=rho, a1=a1),
                self.astrometric_shift_at(position, rho=rho, a1=a1),
            )
        return self._solver.joint(position, rho, a1=a1 if self.source == "linear" else 0.0)

    def joint(self, times, params):
        """Return magnification and source-relative shift for one trajectory."""
        return self.joint_at(self.trajectory(times, params), rho=params.rho, a1=params.a1)

    def astrometric_shift(self, times, params):
        """Complex source-relative shift in the ``tau + 1j*beta`` trajectory frame.

        Results preserve ``times.shape`` and are in angular Einstein-radius
        units, not RA/Dec. This models the unresolved images of a dark lens;
        angular scaling, sky rotation, source proper motion/parallax, and
        blended light are outside this method.
        """
        return self.astrometric_shift_at(
            self.trajectory(times, params), rho=params.rho, a1=params.a1
        )

    def flux(self, times, params, *, f_s=1.0, f_b=0.0):
        """Observed flux with scalar coefficients: ``f_s*magnification + f_b``."""
        if jnp.ndim(f_s) != 0 or jnp.ndim(f_b) != 0:
            raise ValueError("f_s and f_b must be scalar for one dataset; use jax.vmap to batch.")
        return f_s * self.magnification(times, params) + f_b

    def fit_flux(self, times, params, observed_flux, flux_err):
        """Fit source/blend fluxes for one dataset; return a named ``FluxFit``.

        Event parameters are held fixed while fitting the two coefficients.
        Reported flux errors assume those event parameters are known. For HMC,
        either sample fluxes with explicit priors or use ``marginalize_flux``
        with Gaussian priors.
        """
        return fit_flux(self.magnification(times, params), observed_flux, flux_err)

    def marginalize_flux(
        self, times, params, observed_flux, flux_err, *, prior_mean, prior_covariance
    ):
        """Integrate source/blend fluxes under a Gaussian prior for one dataset.

        Supply ``prior_mean`` (2,) and ``prior_covariance`` (2, 2) in
        ``(f_s, f_b)`` order, and measurement standard deviations ``flux_err``.
        Returns the normalized log likelihood and conditional flux posterior
        mean/covariance. Gaussian priors allow negative fluxes.
        """
        return marginalize_flux(
            self.magnification(times, params),
            observed_flux,
            flux_err,
            prior_mean=prior_mean,
            prior_covariance=prior_covariance,
        )
