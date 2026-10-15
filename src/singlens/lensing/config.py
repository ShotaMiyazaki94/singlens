"""Static FFT solver configuration."""

import math
import operator
from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class FFTConfig:
    """Numerical settings for finite-source magnification.

    ``n_fft`` sets the FFT resolution; the default is 4096. ``log_u_min`` and
    ``log_u_max`` set the base-10 separation range. ``rho_switch`` selects the
    small-source approximation, whose outer limit is ``u_switch * rho``.
    """

    n_fft: int = 4096
    log_u_min: float = -8
    log_u_max: float = 4
    rho_switch: float = 1e-4
    u_switch: float = 20

    def __post_init__(self):
        try:
            n_fft = operator.index(self.n_fft)
        except TypeError as exc:
            raise ValueError("n_fft must be an even integer >= 64.") from exc
        if n_fft < 64 or n_fft % 2:
            raise ValueError("n_fft must be an even integer >= 64.")
        if not (
            math.isfinite(self.log_u_min)
            and math.isfinite(self.log_u_max)
            and self.log_u_min < self.log_u_max
        ):
            raise ValueError("log_u_min and log_u_max must be finite and increasing.")
        if not (
            math.isfinite(self.rho_switch)
            and self.rho_switch > 0
            and math.isfinite(self.u_switch)
            and self.u_switch > 0
        ):
            raise ValueError("rho_switch and u_switch must be finite and positive.")
