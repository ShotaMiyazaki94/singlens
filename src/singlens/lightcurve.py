"""Event parameters to single-lens magnification."""

from functools import lru_cache

import jax

from .model import Model
from .parameters import Parameters


@lru_cache(maxsize=16)
def _prepared_model(source, parallax, fft, x64):
    # A first call may occur inside jit/grad. Never cache traced arrays, and
    # never reuse a float32 kernel after the application enables float64.
    with jax.ensure_compile_time_eval():
        return Model(source=source, parallax=parallax, fft=fft)


def magnification(
    times,
    t0,
    u0,
    t_e,
    *,
    rho=None,
    a1=None,
    parallax=None,
    pi_e_n=None,
    pi_e_e=None,
    fft=None,
):
    """Return magnification from microlensing parameters, preserving time shape.

    Times, ``t0`` and the positive Einstein timescale ``t_e`` share units
    (days for annual parallax). ``u0`` is signed, in Einstein-radius units.
    Omit ``rho`` for a point source; supply a nonnegative radius for a uniform
    disk. Adding ``a1`` selects ``I(mu) = 1 - a1*(1-mu)``, with ``0 <= a1 <= 1``.

    For annual parallax, prepare an :class:`AnnualParallax` outside JIT and
    pass both north/east amplitudes ``pi_e_n`` and ``pi_e_e``. Capture geometry
    and optional :class:`FFTConfig` settings in a closure when using JIT.
    Parameters are scalar per event; use ``jax.vmap`` for multiple events.

    Enable JAX float64 before finite-source calls. :class:`Model` also accepts
    named parameters and evaluates magnification at source positions.
    """
    source = "point" if rho is None else ("uniform" if a1 is None else "linear")
    model = _prepared_model(source, parallax, fft, jax.config.x64_enabled)
    params = Parameters(t0=t0, u0=u0, t_e=t_e, rho=rho, a1=a1, pi_e_n=pi_e_n, pi_e_e=pi_e_e)
    return model.magnification(times, params)
