"""Prepared annual-parallax geometry for the high-level model API."""

import math
from dataclasses import dataclass, field

import jax.numpy as jnp

from .ephemeris import (
    EarthOrbitalParallaxProjector,
    HeliocentricEphemeris,
    earth_orbital_parallax_offsets,
    load_builtin_earth_ephemeris,
)


@dataclass(frozen=True, kw_only=True, eq=False)
class AnnualParallax:
    """Sky position, reference epoch, and time convention for annual parallax.

    ``ra_deg`` and ``dec_deg`` are ICRS degrees. Observation/reference times are
    interpreted as ``time + time_origin``; the default origin is JD 2450000.
    Set ``time_origin=0`` for absolute dates. ``use_hjd=True`` takes HJD;
    ``False`` takes JD TDB. The default uses the bundled Earth ephemeris;
    supply ``ephemeris`` to use a custom ``HeliocentricEphemeris``.

    The fitted parallax amplitudes belong in ``Parameters``, not this object.
    """

    ra_deg: float
    dec_deg: float
    t_ref: float
    time_origin: float = 2_450_000.0
    use_hjd: bool = True
    ephemeris: HeliocentricEphemeris | None = field(default=None, repr=False)
    _projector: EarthOrbitalParallaxProjector = field(init=False, repr=False)

    def __post_init__(self):
        if not all(
            math.isfinite(v) for v in (self.ra_deg, self.dec_deg, self.t_ref, self.time_origin)
        ):
            raise ValueError("Sky coordinates, t_ref, and time_origin must be finite.")
        if not -90 <= self.dec_deg <= 90:
            raise ValueError("dec_deg must be between -90 and 90 degrees.")
        eph = self.ephemeris if self.ephemeris is not None else load_builtin_earth_ephemeris()
        projector = EarthOrbitalParallaxProjector(
            eph,
            self.ra_deg,
            self.dec_deg,
            self.t_ref + self.time_origin,
            use_HJD=self.use_hjd,
        )
        object.__setattr__(self, "_projector", projector)

    def offsets(self, times, *, pi_e_n, pi_e_e):
        """Return ``(delta_tau, delta_beta)``, each with the shape of ``times``."""
        times = jnp.asarray(times)
        dtau, dbeta = earth_orbital_parallax_offsets(
            times.reshape(-1) + self.time_origin, pi_e_n, pi_e_e, self._projector
        )
        return dtau.reshape(times.shape), dbeta.reshape(times.shape)
