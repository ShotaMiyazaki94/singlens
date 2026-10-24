"""Experimental finite-source solvers using concentric uniform disks.

The normalized linear limb-darkening bases obey the exact identity
``A_mu = 3/2 * integral_0^1 (1-mu**2) A_uniform(rho*sqrt(1-mu**2)) dmu``.
A change of variable ``mu=sin(phi)`` removes its small-disk endpoint.
For a point lens, the only interior contact is known analytically and the
integration rule can split there. All quadrature orders are fixed under JIT.

These solvers do not change JAX precision settings. Enable float64 before
construction for high accuracy. At exact source-limb contact, derivatives of
the uniform component are logarithmically singular: a finite-order quadrature
returns an order-dependent finite value, not the mathematical derivative.
"""

import operator

import jax.numpy as jnp
import numpy as np

from .point import point_source_shift, pspl


def _check_order(n, name):
    if isinstance(n, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        n = operator.index(n)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if n < 4:
        raise ValueError(f"{name} must be at least 4.")
    return n


def _unit_rule(n):
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return (nodes + 1) / 2, weights / 2


def _angular_rule(n):
    """Endpoint-graded Gauss nodes and weights for angles in [0, pi/2]."""
    x, w = _unit_rule(n)
    # Resolve the angular boundary layer as the source limb approaches u.
    return jnp.asarray(np.pi / 2 * (1 - (1 - x) ** 2)), jnp.asarray(w * np.pi * (1 - x))


def _disk_average(u, rho, nodes, weights, *, order=None):
    """Broadcasting polar disk average; ``rho`` must be positive.

    ``nodes`` and ``weights`` integrate an angle over [0, pi/2]. ``order=None``
    uses the exact point-lens radial primitive. Orders 0, 2, 4 integrate the
    successive small-separation expansions, for experimental Gould corrections.
    The radial integral is analytic, so only one angular quadrature remains.
    """
    if order not in (None, 0, 2, 4):
        raise ValueError("order must be None, 0, 2, or 4.")
    if order is None:
        # One exact flux kernel serves photometry and joint astrometry. JIT
        # removes the unused moment calculations for a photometric-only call.
        return _disk_moments(u, rho, nodes, weights)[0]
    u, rho = jnp.broadcast_arrays(jnp.asarray(u), jnp.asarray(rho))
    inside = u <= rho
    ui = jnp.where(inside, u, 0.5 * rho)[..., None]
    uo = jnp.where(inside, 2 * rho, u)[..., None]
    r = rho[..., None]
    # A node extremely close to pi/2 can round across that endpoint in
    # float32. Keep positive ray lengths even at that reduced precision.
    c = jnp.maximum(jnp.cos(nodes), jnp.finfo(nodes.dtype).eps / 2)

    # Opposite rays inside the disk. The product of their lengths is rho²-u².
    # (rho-u)*(rho+u) preserves precision arbitrarily close to contact.
    d = jnp.sqrt((r - ui) * (r + ui) + (ui * c) ** 2)
    plus = d + ui * c
    minus = (r - ui) * (r + ui) / plus

    # Outside, sin(theta)=rho/u*sin(t) removes the square-root ray endpoint.
    k = r / uo
    q = jnp.sqrt((1 - k) * (1 + k) + (k * c) ** 2)
    outer_plus = uo * q + r * c
    outer_minus = (uo - r) * (uo + r) / outer_plus

    def primitive(x):
        result = x
        if order >= 2:
            result = result + x**3 / 8
        if order >= 4:
            result = result - x**5 / 128
        return result

    inner_integrand = 2 * (primitive(plus) + primitive(minus)) / (jnp.pi * r**2)
    a, b = outer_plus, outer_minus
    difference = jnp.ones_like(a)
    if order >= 2:
        difference = difference + (a**2 + a * b + b**2) / 8
    if order >= 4:
        difference = difference - (a**4 + a**3 * b + a**2 * b**2 + a * b**3 + b**4) / 128
    outer_integrand = 4 / (jnp.pi * uo) * c**2 / q * difference
    values = jnp.where(inside[..., None], inner_integrand, outer_integrand)
    return jnp.sum(weights * values, axis=-1)


def _inputs(u, rho):
    raw_u = jnp.asarray(u)
    # abs has a nonzero autodiff convention at zero. A radial finite source
    # instead has an exactly zero first derivative at alignment.
    u = jnp.atleast_1d(jnp.where(raw_u == 0, 0.0, jnp.abs(raw_u)))
    rho = jnp.asarray(rho)
    if rho.ndim != 0:
        raise ValueError("rho must be scalar; use jax.vmap for multiple radii.")
    if jnp.iscomplexobj(rho):
        raise ValueError("rho must be real.")
    valid = jnp.isfinite(rho) & (rho >= 0)
    safe_rho = jnp.where(valid & (rho > 0), rho, 1.0)
    return u, rho, safe_rho, valid


def _excess_moment_difference(a, b):
    """Stable divided difference of H(s)=G(s)-s**3/3-G(0).

    G(s)=hypot(s,2)*(s**2+1)/3 integrates the image first moment.
    H integrates its excess above the unlensed source position. Factoring
    the divided difference into positive bounded ratios avoids subtracting
    large primitives in the wings and nearly equal ones at the center.
    """
    ha, hb = jnp.hypot(a, 2.0), jnp.hypot(b, 2.0)
    da, db = ha + a, hb + b
    t = (a + b) / (ha + hb)
    p = (ha / da) * (hb / db) + (2 / da) * (2 / db)
    return (4 * (t * p + (a / da) * (b / db)) + t) / 3


def _disk_moments(u, rho, nodes, weights):
    """Return flux F, B=M/u-F, and N=M-u*F for a positive-radius disk.

    M is its unnormalized radial image moment. B has a regular central limit;
    N retains the 1/u far tail even where B itself underflows. The flux and
    excess moment share a single angular rule and ray geometry. No small-
    source, central-series, or far-field approximation is used.
    """
    u, rho = jnp.broadcast_arrays(jnp.asarray(u), jnp.asarray(rho))
    inside = u <= rho
    ui = jnp.where(inside, u, 0.5 * rho)[..., None]
    uo = jnp.where(inside, 2 * rho, u)[..., None]
    r = rho[..., None]
    c = jnp.maximum(jnp.cos(nodes), jnp.finfo(nodes.dtype).eps / 2)
    d = jnp.sqrt((r - ui) * (r + ui) + (ui * c) ** 2)
    a = d + ui * c
    b = (r - ui) * (r + ui) / a
    # E(s)=2*s/(hypot(s,2)+s) integrates (A_point-1)*s.
    ea = 2 * a / (jnp.hypot(a, 2.0) + a)
    eb = 2 * b / (jnp.hypot(b, 2.0) + b)
    excess_inside = 2 / (jnp.pi * r**2) * (ea + eb)
    q_inside = 4 / (jnp.pi * r**2) * c**2 * _excess_moment_difference(a, b)

    # Dimensionless ray endpoints avoid forming u**2 in the far wings.
    k = r / uo
    q = jnp.sqrt((1 - k) * (1 + k) + (k * c) ** 2)
    a = uo * (q + k * c)
    b = uo * ((1 - k) * (1 + k) / (q + k * c))
    ha, hb = jnp.hypot(a, 2.0), jnp.hypot(b, 2.0)
    # Rationalized E(a)-E(b), after cancellation with the ray Jacobian.
    inverse = ((1 / ha) * (1 / hb)) / (a / ha + b / hb)
    inverse = (inverse / (ha + a)) / (hb + b)
    excess_outside = 64 / jnp.pi * c**2 * inverse
    q_outside = 4 / jnp.pi * c**2 * _excess_moment_difference(a, b) / uo

    excess = jnp.sum(weights * jnp.where(inside[..., None], excess_inside, excess_outside), axis=-1)
    b_inside = jnp.sum(weights * (q_inside - excess_inside), axis=-1)
    n_outside = jnp.sum(weights * (q_outside - uo * excess_outside), axis=-1)
    safe_u = jnp.where(inside, 1.0, u)
    b_total = jnp.where(inside, b_inside, n_outside / safe_u)
    n_total = jnp.where(inside, u * b_inside, n_outside)
    return 1 + excess, b_total, n_total


def _position_radius(position):
    """Position magnitude with a regular inactive sqrt branch at the origin."""
    position = jnp.asarray(position)
    position = position.astype(jnp.result_type(position, 1.0))
    center = (jnp.real(position) == 0) & (jnp.imag(position) == 0)
    u = jnp.hypot(
        jnp.where(center, 1.0, jnp.real(position)), jnp.where(center, 0.0, jnp.imag(position))
    )
    return position, jnp.where(center, 0.0, u)


def _radial_shift(position, u, flux, coefficient_numerator, radial_numerator):
    """Regular Cartesian center derivative and a stable distant 1/u tail."""
    center = u == 0
    direction = position / jnp.where(center, 1.0, u)
    central = position * (coefficient_numerator / flux)
    radial = direction * (radial_numerator / flux)
    return jnp.where(center, central, radial)


class UniformDiskQuadrature:
    """Exact radial integration plus fixed Gauss angular quadrature.

    Construct once outside JIT. ``A`` returns at least one dimension;
    ``rho`` must be scalar and nonnegative.
    Default 64-point quadrature targets float64 calculations; increasing ``n``
    improves resolution very close to limb contact.
    """

    def __init__(self, n=64):
        self.n = _check_order(n, "n")
        self._nodes, self._weights = _angular_rule(self.n)

    def A(self, u, rho):
        u, rho, safe_rho, valid = _inputs(u, rho)
        values = _disk_average(u, safe_rho, self._nodes, self._weights)
        point = pspl(jnp.where(rho == 0, u, 1.0))
        return jnp.where(valid, jnp.where(rho == 0, point, values), jnp.nan)

    def joint(self, position, rho):
        """Return magnification and source-relative image-centroid shift.

        Both outputs preserve ``position.shape``; signed real and complex
        positions are supported. Shifts are in angular Einstein-radius units.
        Flux and moment share the same uniform-disk angular integration.
        """
        position, u = _position_radius(position)
        _, rho, safe_rho, valid = _inputs(u, rho)
        flux, b, n = _disk_moments(u, safe_rho, self._nodes, self._weights)
        shift = _radial_shift(position, u, flux, b, n)
        flux = jnp.where(rho == 0, pspl(jnp.where(rho == 0, u, 1.0)), flux)
        shift = jnp.where(rho == 0, point_source_shift(position), shift)
        return jnp.where(valid, flux, jnp.nan), jnp.where(valid, shift, jnp.nan)

    def shift(self, position, rho):
        """Source-relative astrometric shift, preserving position shape."""
        return self.joint(position, rho)[1]


class ConcentricDisk:
    """Linear limb darkening from uniform-disk quadrature (experimental).

    ``n_disk`` controls the angular uniform-source rule. ``n_limb`` is the
    number of Gauss points *per subinterval* in the outer disk mixture. When
    ``split=True``, interior contact uses two graded subintervals, hence up to
    ``2*n_limb + 1`` uniform-source evaluations per position. This method has
    no FFT setup and cost scales linearly with the number of positions.

    ``a1`` is the conventional coefficient in ``I(mu)=1-a1*(1-mu)``. The
    normalized mixture coefficient is ``Gamma=2*a1/(3-a1)``. ``basis`` returns
    the uniform and normalized pure-mu magnifications, reusable for fitting
    several limb-darkening coefficients at fixed geometry.
    """

    def __init__(self, n_disk=64, n_limb=32, split=True, a1=0.5):
        self.n_disk = _check_order(n_disk, "n_disk")
        self.n_limb = _check_order(n_limb, "n_limb")
        self.split = bool(split)
        self.a1 = a1
        self._nodes, self._weights = _angular_rule(self.n_disk)
        x, w = _unit_rule(self.n_limb)
        self._limb_nodes, self._limb_weights = jnp.asarray(x), jnp.asarray(w)

    def _disk(self, u, rho):
        """Uniform disk hook for alternative radial primitives."""
        return _disk_average(u, rho, self._nodes, self._weights)

    def _limb_rule(self, u, rho):
        x, w = self._limb_nodes, self._limb_weights
        if self.split:
            interior = u < rho
            # Keep the unused acos branch away from its divergent derivative.
            z = jnp.where(interior, u / rho, 0.5)
            contact = jnp.where(interior, jnp.arccos(z), jnp.pi / 4)
            # Cubic grading makes the contact logarithm integrable with rapidly
            # convergent fixed-order Gauss quadrature, including its gradients.
            left = contact[..., None] * (1 - (1 - x) ** 3)
            right = contact[..., None] + (jnp.pi / 2 - contact[..., None]) * x**3
            wl = w * 3 * contact[..., None] * (1 - x) ** 2
            wr = w * 3 * (jnp.pi / 2 - contact[..., None]) * x**2
            phi = jnp.concatenate((left, right), axis=-1)
            weights = jnp.concatenate((wl, wr), axis=-1)
        else:
            phi = jnp.pi / 2 * x
            weights = jnp.pi / 2 * w
        cosine = jnp.maximum(jnp.cos(phi), jnp.finfo(phi.dtype).eps / 2)
        return cosine, 1.5 * weights * cosine**3

    def _limb(self, u, rho):
        cosine, weights = self._limb_rule(u, rho)
        disks = self._disk(u[..., None], rho * cosine)
        return jnp.sum(weights * disks, axis=-1)

    def _joint_integrals(self, u, rho):
        uniform = _disk_moments(u, rho, self._nodes, self._weights)
        cosine, weights = self._limb_rule(u, rho)
        disks = _disk_moments(u[..., None], rho * cosine, self._nodes, self._weights)
        limb = tuple(jnp.sum(weights * value, axis=-1) for value in disks)
        return uniform, limb

    def joint_basis(self, u, rho):
        """Return normalized ``(F0, Fmu, C0, Cmu)`` photometric/shift bases.

        ``Cj=shift_j/u`` has a finite central limit. For a mixture, its shift
        coefficient is ``((1-Gamma)*F0*C0+Gamma*Fmu*Cmu)/F``. This is a
        flux-weighted average; averaging the shifts directly is incorrect.
        Scalar inputs return one-dimensional arrays, like :meth:`basis`.
        Use :meth:`joint` for very large positions: it also retains a radial
        numerator so the 1/u shift survives when the 1/u**2 coefficient
        underflows. The zero-radius central coefficient is 1/2.
        """
        u, rho, safe_rho, valid = _inputs(u, rho)
        (f0, b0, _), (fm, bm, _) = self._joint_integrals(u, safe_rho)
        point = pspl(jnp.where(rho == 0, u, 1.0))
        point_coefficient = (1 / jnp.hypot(u, jnp.sqrt(2.0))) ** 2
        c0, cm = b0 / f0, bm / fm
        return tuple(
            jnp.where(
                valid, jnp.where(rho == 0, point if i < 2 else point_coefficient, value), jnp.nan
            )
            for i, value in enumerate((f0, fm, c0, cm))
        )

    def joint(self, position, rho, *, a1=None):
        """Return magnification and shift from one shared disk integration.

        Outputs preserve ``position.shape``, including signed real and complex
        positions. Shifts are relative to the unlensed source center and are
        in angular Einstein-radius units. No approximate-regime switches are
        used. Exact alignment has the correct finite Cartesian Jacobian.
        """
        position, u = _position_radius(position)
        _, rho, safe_rho, valid = _inputs(u, rho)
        coefficient = jnp.asarray(self.a1 if a1 is None else a1)
        valid = valid & jnp.isfinite(coefficient) & (coefficient >= 0) & (coefficient <= 1)
        coefficient = jnp.where(valid, coefficient, 0.0)
        gamma = 2 * coefficient / (3 - coefficient)
        uniform, limb = self._joint_integrals(u, safe_rho)
        flux, b, n = tuple((1 - gamma) * disk + gamma * mu for disk, mu in zip(uniform, limb))
        shift = _radial_shift(position, u, flux, b, n)
        flux = jnp.where(rho == 0, pspl(jnp.where(rho == 0, u, 1.0)), flux)
        shift = jnp.where(rho == 0, point_source_shift(position), shift)
        return jnp.where(valid, flux, jnp.nan), jnp.where(valid, shift, jnp.nan)

    def shift(self, position, rho, *, a1=None):
        """Source-relative astrometric shift, preserving position shape."""
        return self.joint(position, rho, a1=a1)[1]

    def basis(self, u, rho):
        """Return ``(A_uniform, A_mu)`` with the same output shape as ``A``."""
        u, rho, safe_rho, valid = _inputs(u, rho)
        uniform, limb = self._disk(u, safe_rho), self._limb(u, safe_rho)
        point = pspl(jnp.where(rho == 0, u, 1.0))
        uniform = jnp.where(valid, jnp.where(rho == 0, point, uniform), jnp.nan)
        limb = jnp.where(valid, jnp.where(rho == 0, point, limb), jnp.nan)
        return uniform, limb

    def A(self, u, rho, *, a1=None):
        """Evaluate at ``u`` with a scalar radius and optional dynamic ``a1``."""
        coefficient = jnp.asarray(self.a1 if a1 is None else a1)
        valid = jnp.isfinite(coefficient) & (coefficient >= 0) & (coefficient <= 1)
        coefficient = jnp.where(valid, coefficient, 0.0)
        gamma = 2 * coefficient / (3 - coefficient)
        uniform, limb = self.basis(u, rho)
        # Avoid 0*inf for the zero-radius point-source limit at alignment,
        # including a1=0 and a1=1, and keep its coefficient derivative zero.
        point_limit = jnp.asarray(rho) == 0
        safe_uniform = jnp.where(point_limit, 1.0, uniform)
        safe_limb = jnp.where(point_limit, 1.0, limb)
        mixed = (1 - gamma) * safe_uniform + gamma * safe_limb
        return jnp.where(valid, jnp.where(point_limit, uniform, mixed), jnp.nan)
