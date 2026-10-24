"""Offline uniform/mu basis tables with JAX cubic interpolation.

Table construction requires SciPy; evaluation needs only JAX and NumPy. Tables
have an explicit finite domain, and do not extrapolate. Interpolation error,
especially derivatives at source contact, must be measured for each grid.
"""

import json
from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .point import point_source_shift, pspl


@lru_cache(maxsize=2)
def _bundled_basis_table(x64):
    """Share the packaged table, with separate caches for JAX precision modes."""
    # The x64 argument is a cache key: never reuse float32 coefficients after
    # enabling x64. A cold call inside jit must not retain tracers in the cache.
    with jax.ensure_compile_time_eval():
        resource = files("singlens.data").joinpath("finite_source_table.npz")
        with resource.open("rb") as stream:
            return BasisTable.load(stream)


def contact_grid():
    """A nonuniform separation/radius grid with knots on both sides of contact."""
    offsets = np.geomspace(1e-6, 0.2, 80)
    return np.unique(
        np.r_[np.linspace(0, 0.8, 65), 1 - offsets, 1.0, 1 + offsets, np.geomspace(1.2, 30, 90)]
    )


def joint_grid(refinement=1):
    """Contact-resolving grid including long astrometric wings, through z=1e8."""
    if not isinstance(refinement, int) or isinstance(refinement, bool) or refinement < 1:
        raise ValueError("refinement must be a positive integer.")
    grid = np.unique(np.r_[contact_grid(), np.geomspace(30, 1e8, 161)])
    fractions = np.arange(refinement) / refinement
    return np.r_[(grid[:-1, None] + np.diff(grid)[:, None] * fractions).ravel(), grid[-1]]


def _axis_weights(value, knots):
    """Four nonzero cubic B-splines (Cox-de Boor), including closed endpoints."""
    span = jnp.clip(jnp.searchsorted(knots, value, side="right") - 1, 3, len(knots) - 5)
    weights = [jnp.ones_like(value)]
    for degree in range(1, 4):
        saved = jnp.zeros_like(value)
        next_weights = []
        for r in range(degree):
            right = knots[span + r + 1] - value
            left = value - knots[span + r + 1 - degree]
            term = weights[r] / (right + left)
            next_weights.append(saved + right * term)
            saved = left * term
        weights = next_weights + [saved]
    return span[..., None] + jnp.arange(-3, 1), jnp.stack(weights, axis=-1)


class BasisTable:
    """Interpolated uniform/mu flux bases, optionally with astrometric moments.

    ``Model`` automatically loads the bundled joint table for uniform and
    linear sources. Load a custom table with :meth:`load` and pass it to
    ``Model(source=..., solver=table)``. A table needs four basis channels
    for astrometry; photometry-only tables have two.

    Cubic B-splines interpolate the stored bases. Values outside the stored
    domain return NaN; ``rho=0`` gives the analytic point-source limit.
    The physical singular slope at exact contact is smoothed by interpolation.
    Table construction is an offline operation requiring SciPy.
    """

    def __init__(
        self,
        rho_grid,
        z_grid,
        knots_rho,
        knots_z,
        coefficients,
        *,
        a1=0.5,
        coordinate="ratio",
        representation="raw",
        metadata=None,
    ):
        self.rho_grid = np.asarray(rho_grid, dtype=float)
        self.z_grid = np.asarray(z_grid, dtype=float)
        for name, grid in (("rho_grid", self.rho_grid), ("z_grid", self.z_grid)):
            if (
                grid.ndim != 1
                or grid.size < 4
                or not np.all(np.isfinite(grid))
                or not np.all(np.diff(grid) > 0)
            ):
                raise ValueError(f"{name} must have at least four finite increasing values.")
        if self.rho_grid[0] <= 0 or self.z_grid[0] < 0:
            raise ValueError("rho_grid must be positive and z_grid nonnegative.")
        knots_rho, knots_z = np.asarray(knots_rho), np.asarray(knots_z)
        coefficients = np.asarray(coefficients)
        if (
            coefficients.ndim != 3
            or coefficients.shape[:2] != (len(self.rho_grid), len(self.z_grid))
            or coefficients.shape[-1] not in (2, 4)
            or not np.all(np.isfinite(coefficients))
        ):
            raise ValueError("coefficients must be finite with shape (n_rho, n_z, 2 or 4).")
        if coordinate not in ("ratio", "log1p_ratio2"):
            raise ValueError("Unknown table separation coordinate.")
        self.has_astrometry = coefficients.shape[-1] == 4
        if self.has_astrometry and coordinate != "log1p_ratio2":
            raise ValueError("Astrometric tables require the even log1p_ratio2 coordinate.")
        if representation not in ("raw", "point_residual"):
            raise ValueError("Unknown table basis representation.")
        if representation == "point_residual" and not self.has_astrometry:
            raise ValueError("Point residuals require four astrometric basis channels.")
        self.coordinate = coordinate
        self.representation = representation
        self.metadata = {} if metadata is None else dict(metadata)
        transformed_z = self.z_grid if coordinate == "ratio" else np.log1p(self.z_grid**2)
        for knots, grid in ((knots_rho, np.log(self.rho_grid)), (knots_z, transformed_z)):
            if (
                knots.shape != (len(grid) + 4,)
                or not np.all(np.isfinite(knots))
                or not np.all(np.diff(knots) >= 0)
                or not np.allclose(knots[:4], grid[0], rtol=0, atol=1e-14)
                or not np.allclose(knots[-4:], grid[-1], rtol=0, atol=1e-14)
                or np.any(np.diff(knots[3:-3]) <= 0)
            ):
                raise ValueError("Invalid clamped cubic spline knots.")
        # Keep the validated host knots for serialization. JAX may round them
        # to float32, while the domain grids above retain their original dtype.
        # Saving rounded knots alongside those grids can produce an invalid
        # archive whose clamped endpoints no longer match its domain.
        self._host_knots_rho = np.array(knots_rho, copy=True)
        self._host_knots_z = np.array(knots_z, copy=True)
        self._knots_rho = jnp.asarray(self._host_knots_rho)
        self._knots_z = jnp.asarray(self._host_knots_z)
        self._coefficients = jnp.asarray(coefficients)
        self.a1 = a1

    @classmethod
    def build(
        cls,
        solver,
        rho_grid=None,
        z_grid=None,
        *,
        batch_size=64,
        a1=0.5,
        astrometry=False,
    ):
        """Precompute bases using a solver's ``basis(u, rho)`` method.

        Photometry defaults are 96 radii from 1e-5 to 10 and :func:`contact_grid`.
        With astrometry, defaults are 192 radii from 1e-6 to 10 and 947 ratios
        from ``joint_grid(refinement=2)``; the solver must implement ``joint_basis`` returning
        ``(F_uniform, F_mu, C_uniform, C_mu)`` with ``C=shift/u``.
        Validate values and derivatives independently when changing these grids.
        Enable JAX float64 first. SciPy is imported only during this offline operation.
        """
        from scipy.interpolate import RectBivariateSpline

        lower = 1e-6 if astrometry else 1e-5
        n_rho = 192 if astrometry else 96
        rho_grid = np.asarray(np.geomspace(lower, 10, n_rho) if rho_grid is None else rho_grid)
        if z_grid is None:
            z_grid = joint_grid(refinement=2) if astrometry else contact_grid()
        z_grid = np.asarray(z_grid)
        for name, grid in (("rho_grid", rho_grid), ("z_grid", z_grid)):
            if (
                grid.ndim != 1
                or grid.size < 4
                or not np.all(np.isfinite(grid))
                or not np.all(np.diff(grid) > 0)
            ):
                raise ValueError(f"{name} must have at least four finite increasing values.")
        if rho_grid[0] <= 0 or z_grid[0] < 0:
            raise ValueError("rho_grid must be positive and z_grid nonnegative.")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        rr, zz = np.meshgrid(rho_grid, z_grid, indexing="ij")
        radii, positions = rr.ravel(), (rr * zz).ravel()
        n_bases = 4 if astrometry else 2
        basis = solver.joint_basis if astrometry else solver.basis
        evaluate = jax.jit(jax.vmap(lambda u, r: jnp.stack(basis(u, r)).reshape(n_bases)))
        blocks = []
        for start in range(0, len(radii), batch_size):
            stop = min(start + batch_size, len(radii))
            pad = batch_size - (stop - start)
            r = np.pad(radii[start:stop], (0, pad), mode="edge")
            u = np.pad(positions[start:stop], (0, pad), mode="edge")
            blocks.append(np.asarray(evaluate(u, r))[: stop - start])
        values = np.concatenate(blocks).reshape(len(rho_grid), len(z_grid), n_bases)
        if not np.all(np.isfinite(values)) or not np.all(values[..., :2] > 0):
            raise ValueError("The generating solver returned invalid flux or moment bases.")
        transformed = values.copy()
        if astrometry:
            # Store only finite-source corrections. Interpolating the full
            # point-source dependence on u=rho*z produces large, cancelling
            # coordinate derivatives at fixed u as rho tends to zero.
            # R=hypot(u,rho) gives a finite flux baseline even at alignment.
            scale = np.hypot(rr * zz, rr)
            h = np.hypot(scale, 2.0)
            baseline_flux = scale / h + (2 / h) / scale
            baseline_coefficient = 1 / ((rr * zz) ** 2 + 2)
            q = (1 + zz**2)[..., None]
            transformed[..., :2] = q * np.log1p(
                (values[..., :2] - baseline_flux[..., None]) / baseline_flux[..., None]
            )
            transformed[..., 2:] = q * (values[..., 2:] - baseline_coefficient[..., None])
        else:
            transformed[..., :2] = np.log(values[..., :2])
        coordinate = "log1p_ratio2" if astrometry else "ratio"
        transformed_z = np.log1p(z_grid**2) if astrometry else z_grid
        splines = [
            RectBivariateSpline(np.log(rho_grid), transformed_z, transformed[..., b], s=0)
            for b in range(n_bases)
        ]
        tx, ty = splines[0].get_knots()
        coeff = np.stack(
            [s.get_coeffs().reshape(len(rho_grid), len(z_grid)) for s in splines], axis=-1
        )
        metadata = dict(generator=type(solver).__name__, dtype=str(values.dtype))
        for attr in ("n_disk", "n_limb", "split"):
            if hasattr(solver, attr):
                metadata[attr] = getattr(solver, attr)
        return cls(
            rho_grid,
            z_grid,
            tx,
            ty,
            coeff,
            a1=a1,
            coordinate=coordinate,
            representation="point_residual" if astrometry else "raw",
            metadata=metadata,
        )

    def _interpolate(self, radius2, rho):
        """Interpolate all stored channels once, with safe out-of-domain inputs."""
        rho = jnp.asarray(rho)
        if rho.ndim != 0:
            raise ValueError("rho must be scalar; use jax.vmap for multiple radii.")
        if jnp.iscomplexobj(rho):
            raise ValueError("rho must be real.")
        safe_rho = jnp.where(jnp.isfinite(rho) & (rho > 0), rho, self.rho_grid[0])
        ratio2 = radius2 / safe_rho**2
        # Forming |position|**2/rho**2 can round an exact domain endpoint a
        # few ULPs outside, especially for complex positions. Accept only that
        # arithmetic roundoff; the interpolation coordinate is clamped below.
        roundoff = 4 * jnp.finfo(ratio2.dtype).eps
        valid = (
            (rho >= self.rho_grid[0])
            & (rho <= self.rho_grid[-1])
            & (ratio2 >= self.z_grid[0] ** 2 * (1 - roundoff))
            & (ratio2 <= self.z_grid[-1] ** 2 * (1 + roundoff))
        )
        # Strict comparisons preserve the interior one-sided spline derivative
        # at closed domain endpoints (clip would return half that derivative).
        x = jnp.log(safe_rho)
        x = jnp.where(x < self._knots_rho[3], self._knots_rho[3], x)
        x = jnp.where(x > self._knots_rho[-4], self._knots_rho[-4], x)
        if self.coordinate == "log1p_ratio2":
            y = jnp.log1p(ratio2)
        else:
            y = jnp.sqrt(jnp.where(radius2 == 0, 1.0, ratio2))
            y = jnp.where(radius2 == 0, 0.0, y)
        y = jnp.where(y < self._knots_z[3], self._knots_z[3], y)
        y = jnp.where(y > self._knots_z[-4], self._knots_z[-4], y)
        ix, wx = _axis_weights(x, self._knots_rho)
        iy, wy = _axis_weights(y, self._knots_z)
        if radius2.size >= 256:
            # Every epoch shares rho. Contract its four spline rows once,
            # then gather only four separation coefficients per epoch.
            radial = jnp.sum(self._coefficients[ix] * wx[:, None, None], axis=0)
            result = jnp.sum(radial[iy] * wy[..., :, None], axis=-2)
        else:
            c = self._coefficients[ix[:, None], iy[..., None, :]]
            result = jnp.sum(c * wx[:, None, None] * wy[..., None, :, None], axis=(-3, -2))
        return result, valid

    def _decode(self, stored, radius2, rho):
        """Recover fluxes and signed shift coefficients from spline channels."""
        if self.representation == "raw":
            return jnp.concatenate((jnp.exp(stored[..., :2]), stored[..., 2:]), axis=-1)
        safe_rho = jnp.where(jnp.isfinite(rho) & (rho > 0), rho, self.rho_grid[0])
        inverse_q = 1 / (1 + radius2 / safe_rho**2)
        baseline_flux = pspl(jnp.sqrt(radius2 + safe_rho**2))
        fluxes = baseline_flux[..., None] * jnp.exp(stored[..., :2] * inverse_q[..., None])
        coefficients = 1 / (radius2[..., None] + 2) + stored[..., 2:] * inverse_q[..., None]
        return jnp.concatenate((fluxes, coefficients), axis=-1)

    def basis(self, u, rho):
        """Return uniform and normalized pure-mu flux bases (at least 1D)."""
        u = jnp.atleast_1d(jnp.asarray(u))
        table_u = jnp.where(jnp.asarray(rho) == 0, 0.0, u)
        radius2 = jnp.real(table_u) ** 2 + jnp.imag(table_u) ** 2
        stored, valid = self._interpolate(radius2, rho)
        result = self._decode(stored, radius2, rho)[..., :2]
        point = pspl(jnp.where(rho == 0, u, 1.0))
        return tuple(
            jnp.where(rho == 0, point, jnp.where(valid, result[..., b], jnp.nan)) for b in range(2)
        )

    def joint_basis(self, u, rho):
        """Return ``(F0, Fmu, C0, Cmu)``; ``C=shift/u`` has a finite center limit."""
        if not self.has_astrometry:
            raise ValueError("Build a table with astrometry=True to evaluate astrometry.")
        u = jnp.atleast_1d(jnp.asarray(u))
        table_u = jnp.where(jnp.asarray(rho) == 0, 0.0, u)
        radius2 = jnp.real(table_u) ** 2 + jnp.imag(table_u) ** 2
        stored, valid = self._interpolate(radius2, rho)
        decoded = self._decode(stored, radius2, rho)
        point = pspl(jnp.where(rho == 0, u, 1.0))
        fluxes = tuple(
            jnp.where(rho == 0, point, jnp.where(valid, decoded[..., b], jnp.nan)) for b in range(2)
        )
        point_h = jnp.hypot(jnp.abs(u), jnp.sqrt(2.0))
        coefficients = tuple(
            jnp.where(rho == 0, (1 / point_h) / point_h, jnp.where(valid, decoded[..., b], jnp.nan))
            for b in range(2, 4)
        )
        return fluxes + coefficients

    def joint(self, position, rho, *, a1=None):
        """Return magnification and source-relative shift, preserving position shape."""
        if not self.has_astrometry:
            raise ValueError("Build a table with astrometry=True to evaluate astrometry.")
        position = jnp.asarray(position)
        table_position = jnp.where(jnp.asarray(rho) == 0, 0.0, position)
        radius2 = jnp.real(table_position) ** 2 + jnp.imag(table_position) ** 2
        stored, valid = self._interpolate(radius2, rho)
        decoded = self._decode(stored, radius2, rho)
        coefficient = jnp.asarray(self.a1 if a1 is None else a1)
        valid = valid & jnp.isfinite(coefficient) & (coefficient >= 0) & (coefficient <= 1)
        valid_coefficient = jnp.isfinite(coefficient) & (coefficient >= 0) & (coefficient <= 1)
        safe_a1 = jnp.where(valid_coefficient, coefficient, 0.0)
        gamma = 2 * safe_a1 / (3 - safe_a1)
        f0, fm = decoded[..., 0], decoded[..., 1]
        w0, wm = (1 - gamma) * f0, gamma * fm
        flux = w0 + wm
        shift = position * (w0 * decoded[..., 2] + wm * decoded[..., 3]) / flux
        point_flux = pspl(jnp.where(rho == 0, position, 1.0))
        flux = jnp.where(rho == 0, point_flux, jnp.where(valid, flux, jnp.nan))
        shift = jnp.where(rho == 0, point_source_shift(position), jnp.where(valid, shift, jnp.nan))
        return jnp.where(valid_coefficient, flux, jnp.nan), jnp.where(
            valid_coefficient, shift, jnp.nan
        )

    def shift(self, position, rho, *, a1=None):
        """Astrometric shift in Einstein-radius units; signed real or complex."""
        return self.joint(position, rho, a1=a1)[1]

    def A(self, u, rho, *, a1=None):
        """Evaluate linear limb darkening using the two interpolated bases."""
        a1 = jnp.asarray(self.a1 if a1 is None else a1)
        valid = jnp.isfinite(a1) & (a1 >= 0) & (a1 <= 1)
        a1 = jnp.where(valid, a1, 0.0)
        gamma = 2 * a1 / (3 - a1)
        disk, limb = self.basis(u, rho)
        point_limit = jnp.asarray(rho) == 0
        safe_disk = jnp.where(point_limit, 1.0, disk)
        safe_limb = jnp.where(point_limit, 1.0, limb)
        result = (1 - gamma) * safe_disk + gamma * safe_limb
        result = jnp.where(point_limit, disk, result)
        return jnp.where(valid, result, jnp.nan)

    def save(self, path):
        """Save numeric spline data as an NPZ file (no pickled objects)."""
        np.savez_compressed(
            Path(path),
            format_version=3,
            rho_grid=self.rho_grid,
            z_grid=self.z_grid,
            knots_rho=self._host_knots_rho,
            knots_z=self._host_knots_z,
            coefficients=np.asarray(self._coefficients),
            a1=self.a1,
            coordinate=self.coordinate,
            representation=self.representation,
            metadata=json.dumps(self.metadata),
        )

    @classmethod
    def load(cls, path):
        """Load a table saved by :meth:`save`, validating its numeric layout."""
        with np.load(path, allow_pickle=False) as data:
            version = int(data["format_version"])
            if version not in (1, 2, 3):
                raise ValueError("Unsupported basis table format.")
            return cls(
                data["rho_grid"],
                data["z_grid"],
                data["knots_rho"],
                data["knots_z"],
                data["coefficients"],
                a1=float(data["a1"]),
                coordinate="ratio" if version == 1 else str(data["coordinate"]),
                representation=str(data["representation"]) if "representation" in data else "raw",
                metadata={} if version == 1 else json.loads(str(data["metadata"])),
            )
