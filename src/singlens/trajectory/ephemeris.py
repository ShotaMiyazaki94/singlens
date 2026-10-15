"""Uniform Earth ephemerides and their projection onto the source sky plane.

Ephemeris times are absolute JD TDB; optional light-time correction accepts
HJD at evaluation. Coordinates are ICRS, positions AU, and velocities AU/day.
"""

from importlib import resources

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._geometry import _sky_basis

AU_C_DAY = 0.005775518331436995  # AU/c in days


def load_horizons_vectors_file(path: str) -> np.ndarray:
    """Parse a JPL Horizons cartesian-state table.

    Returns ndarray columns ``[t_jdtdb, x, y, z, vx, vy, vz]`` with times in
    JD TDB days, positions in AU, velocities in AU/day. Lines outside the
    ``$$SOE`` … ``$$EOE`` block are skipped; calendar date and LT/RG/RR fields
    are ignored.
    """
    rows = []
    in_block = False
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not in_block:
                if s.startswith("$$SOE"):
                    in_block = True
                continue
            else:
                if s.startswith("$$EOE"):
                    break
                if not s or s.startswith("*"):
                    continue

                parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                if len(parts) < 8:
                    continue

                try:
                    t = float(parts[0])
                    x = float(parts[2])
                    y = float(parts[3])
                    z = float(parts[4])
                    vx = float(parts[5])
                    vy = float(parts[6])
                    vz = float(parts[7])
                except ValueError:
                    continue

                rows.append((t, x, y, z, vx, vy, vz))

    if not rows:
        raise ValueError("No ephemeris rows parsed. Check file format and $$SOE/$$EOE markers.")
    return np.asarray(rows, dtype=np.float64)


@jax.tree_util.register_pytree_node_class
class HeliocentricEphemeris:
    """Uniform heliocentric ephemeris.

    Attributes
    ----------
    t : jax.Array, shape (N,)
        Absolute times (JD TDB) on a uniform grid.
    r : jax.Array, shape (N, 3)
        Heliocentric position in AU.
    v : jax.Array, shape (N, 3)
        Heliocentric velocity in AU/day.

    Notes
    -----
    ``t`` must be uniformly spaced for ``interp_uniform_linear`` to apply.
    """

    def __init__(self, t: jnp.ndarray, r: jnp.ndarray, v: jnp.ndarray):
        self.t = t
        self.r = r
        self.v = v

    def tree_flatten(self):
        return (self.t, self.r, self.v), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)

    @staticmethod
    def from_horizons_vectors_table(table_np: np.ndarray) -> "HeliocentricEphemeris":
        table_np = np.asarray(table_np, dtype=float)
        if table_np.ndim != 2 or table_np.shape[1] != 7 or len(table_np) < 2:
            raise ValueError("Ephemeris must have shape (N, 7), N >= 2.")
        if not np.isfinite(table_np).all():
            raise ValueError("Ephemeris must contain finite values.")
        table_np = table_np[np.argsort(table_np[:, 0])]
        steps = np.diff(table_np[:, 0])
        if steps[0] <= 0 or not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-9):
            raise ValueError("Ephemeris times must be unique and uniformly spaced.")
        tab = jnp.asarray(table_np)
        t = tab[:, 0]
        r = tab[:, 1:4]
        v = tab[:, 4:7]
        return HeliocentricEphemeris(t, r, v)


def interp_uniform_linear(xq, x0, dt, y):
    """Linear interpolation; out-of-range queries return NaN (also under JIT)."""
    xq = jnp.atleast_1d(xq)
    u = (xq - x0) / dt
    i0 = jnp.floor(u).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, y.shape[0] - 2)
    w = u - i0.astype(u.dtype)
    y0 = y[i0]
    y1 = y[i0 + 1]
    result = y0 + (y1 - y0) * (w[:, None] if y.ndim == 2 else w)
    valid = (u >= 0) & (u <= y.shape[0] - 1)
    return jnp.where(valid[:, None] if y.ndim == 2 else valid, result, jnp.nan)


def light_time_corrected_time(t, t0, dt, rv, n_hat, au_c_day: float = AU_C_DAY, n_iter: int = 5):
    """Iteratively solve for emission time given reception time and light travel."""
    t = jnp.atleast_1d(jnp.asarray(t))
    t_emit = t

    def body(_, t_emit_curr):
        rv_curr = interp_uniform_linear(t_emit_curr, t0, dt, rv)
        r_curr = rv_curr[..., :3]
        lt = jnp.sum(r_curr * n_hat, axis=-1) * au_c_day
        return t - lt

    return lax.fori_loop(0, n_iter, body, t_emit)


@jax.tree_util.register_pytree_node_class
class EarthOrbitalParallaxProjector:
    """Map heliocentric Earth ephemeris to sky-plane offsets.

    Applies optional light-time correction (HJD) and stores reference position
    and velocity at ``tref`` to separate annual parallax from linear motion.
    """

    def __init__(
        self,
        eph: HeliocentricEphemeris,
        RA_deg,
        Dec_deg,
        tref,
        *,
        use_HJD: bool = True,
        light_time_iters: int = 5,
        au_c_day: float = AU_C_DAY,
    ):
        dtype = eph.t.dtype
        self.t0 = eph.t[0]
        self.dt = eph.t[1] - eph.t[0]
        self.tref = jnp.asarray(tref, dtype=dtype)

        self.use_HJD = bool(use_HJD)
        self.light_time_iters = int(light_time_iters)
        self.au_c_day = jnp.asarray(au_c_day, dtype=dtype)

        self.sky_north, self.sky_east, self.n_hat = _sky_basis(RA_deg, Dec_deg, dtype=dtype)

        self.rv = jnp.concatenate([eph.r, eph.v], axis=-1)

        if self.use_HJD:
            tref_eval = light_time_corrected_time(
                self.tref[None],
                self.t0,
                self.dt,
                self.rv,
                self.n_hat,
                au_c_day=self.au_c_day,
                n_iter=self.light_time_iters,
            )[0]
        else:
            tref_eval = self.tref

        rv_ref = interp_uniform_linear(tref_eval[None], self.t0, self.dt, self.rv)[0]
        r_ref, v_ref = rv_ref[:3], rv_ref[3:]

        self.E_ref = -jnp.stack([r_ref @ self.sky_east, r_ref @ self.sky_north])
        self.V_ref = -jnp.stack([v_ref @ self.sky_east, v_ref @ self.sky_north])

    def tree_flatten(self):
        children = (
            self.t0,
            self.dt,
            self.tref,
            self.au_c_day,
            self.sky_north,
            self.sky_east,
            self.n_hat,
            self.rv,
            self.E_ref,
            self.V_ref,
        )
        aux = (self.use_HJD, self.light_time_iters)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        obj = object.__new__(cls)
        (
            obj.t0,
            obj.dt,
            obj.tref,
            obj.au_c_day,
            obj.sky_north,
            obj.sky_east,
            obj.n_hat,
            obj.rv,
            obj.E_ref,
            obj.V_ref,
        ) = children
        (obj.use_HJD, obj.light_time_iters) = aux
        return obj


def earth_orbital_parallax_offsets(t, piEN, piEE, P: EarthOrbitalParallaxProjector):
    """Ephemeris-based ``Δtau`` and ``Δbeta`` offsets (JAX differentiable)."""
    t = jnp.atleast_1d(jnp.asarray(t, dtype=P.tref.dtype))

    if P.use_HJD:
        t_eval = light_time_corrected_time(
            t, P.t0, P.dt, P.rv, P.n_hat, au_c_day=P.au_c_day, n_iter=P.light_time_iters
        )
    else:
        t_eval = t

    rv_t = interp_uniform_linear(t_eval, P.t0, P.dt, P.rv)
    r_t = rv_t[:, :3]

    E_t = -jnp.stack([r_t @ P.sky_east, r_t @ P.sky_north], axis=-1)
    ds = -((P.E_ref[None] - E_t) + P.V_ref[None] * (t - P.tref)[:, None])

    d_tau = piEN * ds[:, 1] + piEE * ds[:, 0]
    d_beta = -piEE * ds[:, 1] + piEN * ds[:, 0]
    return d_tau, d_beta


def load_builtin_earth_ephemeris() -> HeliocentricEphemeris:
    """Load the bundled JPL Horizons Earth ephemeris (uniform JD TDB grid)."""
    try:
        path = resources.files("singlens.data").joinpath("earth_orbital_parallax_table.txt")
    except (FileNotFoundError, ModuleNotFoundError) as exc:
        raise FileNotFoundError("Bundled Horizons ephemeris not found.") from exc
    with resources.as_file(path) as local_path:
        table = load_horizons_vectors_file(local_path)
    return HeliocentricEphemeris.from_horizons_vectors_table(table)
