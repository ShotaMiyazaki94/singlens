"""Independent spherical-cap flux identities and smooth-integral convergence."""

import numpy as np
import pytest

from example.inference.code.active_spots import SpotAudit, cap_quadrature


@pytest.mark.parametrize(
    "x,y,area,a1", [(0.0, 0.0, 0.25, 0.0), (0.5, -0.4, 0.1, 0.4), (0.4, 0.5, 0.02, 1.0)]
)
def test_cap_flux_matches_closed_form_spherical_moments(x, y, area, a1):
    """Projected cap area and linear-limb flux have analytic surface moments."""
    _, weights, info = cap_quadrature(x, y, area, a1=a1, chord_y=2.0, n_radial=8, n_azimuth=32)
    mu_center = np.sqrt(1 - x * x - y * y)
    cosine_radius = np.sqrt(1 - area / mu_center)
    integral_cosine_squared = (1 - cosine_radius**3) / 3
    projected_area = np.pi * area
    mu_squared_moment = (
        2
        * np.pi
        * (
            mu_center**2 * integral_cosine_squared
            + (1 - mu_center**2) / 2 * (1 - cosine_radius - integral_cosine_squared)
        )
    )
    expected = ((1 - a1) * projected_area + a1 * mu_squared_moment) / (np.pi * (1 - a1 / 3))
    np.testing.assert_allclose(weights.sum(), expected, rtol=3e-13, atol=1e-15)
    assert info["area_quadrature_error"] < 1e-14


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_non_crossing_spot_integral_converges_and_preserves_unlensed_flux():
    """Stress the sharpest allowed cap and its independently known far-field limit."""
    audit = SpotAudit(n_fft=256)  # This test uses only the smooth spatial integral.
    caps = [(0.0, 0.57, 0.02)]
    times = audit.t0 + audit.t_e * audit.rho * np.array([-0.1, 0.0, 0.1])
    reference = np.full(len(times), 3.0)
    values = [
        audit.spotted_amplification(times, reference, caps, n_radial=n, n_azimuth=3 * n)[0]
        for n in (32, 64, 128)
    ]
    coarse_error = np.max(abs(values[0] - values[2]))
    fine_error = np.max(abs(values[1] - values[2]))
    # Absolute magnification tolerance, far below the per-exposure flux noise.
    assert fine_error < 1e-8
    assert fine_error < coarse_error / 10
    distant = audit.t0 + audit.t_e * np.array([-1e5, 1e5])
    no_lensing, _ = audit.spotted_amplification(distant, np.ones(2), caps)
    np.testing.assert_allclose(no_lensing, np.ones(2), rtol=0, atol=1e-13)
