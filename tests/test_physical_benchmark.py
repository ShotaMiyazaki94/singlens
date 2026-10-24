"""Physical units and observing assumptions for the quiet-planet benchmark."""

from pathlib import Path

import numpy as np
import pytest


def test_planet_scales_follow_mass_distance_and_angular_source_size():
    from example._physical import planet_configuration

    config = planet_configuration()
    event = config["event"]
    # Independently evaluated SI geometry, including the observer--lens distance.
    au, parsec, solar_radius = 149597870700.0, 3.085677581491367e16, 6.957e8
    grav, light, earth_mass = 6.67430e-11, 299792458.0, 5.972167867791379e24
    ds = config["source_distance_kpc"] * 1000 * parsec
    dl = config["lens_distance_kpc"] * 1000 * parsec
    theta_star = config["source_radius_solar"] * solar_radius / ds
    theta_e = np.sqrt(
        4 * grav * config["lens_mass_earth"] * earth_mass / light**2 * (ds - dl) / (ds * dl)
    )
    radians_to_mas = 180 / np.pi * 3600 * 1000
    np.testing.assert_allclose(config["theta_star_mas"], theta_star * radians_to_mas, rtol=1e-12)
    # 8.144 mas/Msun is a rounded microlensing constant.
    np.testing.assert_allclose(config["theta_e_mas"], theta_e * radians_to_mas, rtol=2e-4)
    np.testing.assert_allclose(
        config["pi_rel_mas"], au * (1 / dl - 1 / ds) * radians_to_mas, rtol=1e-12
    )
    np.testing.assert_allclose(event["rho"], theta_star / theta_e, rtol=2e-4)
    np.testing.assert_allclose(
        event["t_e"], theta_e * radians_to_mas / config["mu_geo_masyr"] * 365.25, rtol=2e-4
    )
    # The source-crossing time is a geometric observable independent of lens mass.
    expected_hours = theta_star * radians_to_mas / config["mu_geo_masyr"] * 365.25 * 24
    np.testing.assert_allclose(event["rho"] * event["t_e"] * 24, expected_hours, rtol=1e-12)
    assert 0 < event["u0"] < event["rho"]
    assert abs(config["source_logg_cgs"] - config["atmosphere_logg_cgs"]) < 0.05


def test_quiet_atmosphere_photon_counts_and_blend_use_consistent_units():
    from example._physical import planet_configuration

    config = planet_configuration()
    root = Path(__file__).resolve().parents[1] / "example/inference/data"
    area = np.genfromtxt(root / "roman_f146_sca01_effective_area.csv", delimiter=",", names=True)
    star = np.genfromtxt(root / "ck04_5000_4000_logg4p5_solar.csv", delimiter=",", names=True)
    wavelength = area["wavelength_micron"] * 1e-6
    surface = np.interp(
        wavelength * 1e10, star["wavelength_angstrom"], star["photosphere_surface_flux"]
    )
    # Convert erg/s/cm²/Angstrom to W/m²/m before an independent photon integral.
    surface *= 1e7
    distance_m = config["source_distance_kpc"] * 1000 * 3.085677581491367e16
    dilution = (config["source_radius_solar"] * 6.957e8 / distance_m) ** 2
    extinction = 10 ** (
        -0.4 * config["extinction_at_1p65_micron_mag"] * (wavelength / 1.65e-6) ** -2
    )
    photon_energy = 6.62607015e-34 * 299792458 / wavelength
    source_rate = np.trapezoid(
        surface * dilution * extinction * area["effective_area_m2"] / photon_energy, wavelength
    )
    ab_spectrum = 3631e-26 * 299792458 / wavelength**2
    ab_rate = np.trapezoid(ab_spectrum * area["effective_area_m2"] / photon_energy, wavelength)
    np.testing.assert_allclose(
        config["source_electrons_per_exposure"],
        source_rate * config["exposure_seconds"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        config["source_magnitude_ab"], -2.5 * np.log10(source_rate / ab_rate), atol=1e-12
    )
    np.testing.assert_allclose(config["ab_zero_electrons_per_second"], ab_rate, rtol=1e-12)
    event = config["event"]
    np.testing.assert_allclose(event["f_s"] + event["f_b"], 1, atol=1e-15)
    np.testing.assert_allclose(
        config["baseline_electrons_per_exposure"] * event["f_s"],
        config["source_electrons_per_exposure"],
        rtol=1e-13,
    )
    # Source and baseline AB magnitudes must encode precisely the injected dilution.
    np.testing.assert_allclose(
        10 ** (-0.4 * (config["source_magnitude_ab"] - config["baseline_magnitude_ab"])),
        event["f_s"],
        rtol=1e-13,
    )


def test_noise_anchor_is_per_exposure_and_floor_is_separate():
    from example._physical import photometric_error, planet_configuration

    config = planet_configuration()
    relative_flux = 10 ** (-0.4 * (config["anchor_magnitude_ab"] - config["baseline_magnitude_ab"]))
    error = photometric_error(relative_flux, config)
    without_floor = np.sqrt(error**2 - (relative_flux * config["fractional_floor"]) ** 2)
    np.testing.assert_allclose(relative_flux / without_floor, 100, rtol=1e-12)
    assert relative_flux / error < 100
    # Subtract the independent fractional floor: variance is affine in counts,
    # with the positive background/read intercept required by the S/N anchor.
    means = np.array([1.0, 2.0, 3.0])
    electron_variance = (
        photometric_error(means, config) ** 2 - (means * config["fractional_floor"]) ** 2
    ) * config["baseline_electrons_per_exposure"] ** 2
    np.testing.assert_allclose(
        np.diff(electron_variance), config["baseline_electrons_per_exposure"], rtol=1e-12
    )
    assert electron_variance[0] > config["baseline_electrons_per_exposure"]


@pytest.mark.parametrize("mean", [0.0, -1.0, np.nan, np.inf])
def test_nonphysical_flux_cannot_generate_measurement_errors(mean):
    from example._physical import photometric_error, planet_configuration

    with pytest.raises(ValueError, match="finite and positive"):
        photometric_error(mean, planet_configuration())


def test_saturation_checks_total_blended_light():
    from example._physical import planet_configuration, saturation_check

    config = planet_configuration()
    event = config["event"]
    source_amplification = np.array([1.0, 3.0])
    combined = event["f_s"] * source_amplification + event["f_b"]
    result = saturation_check(combined, config)
    independent_peak_magnitude = config["source_magnitude_ab"] - 2.5 * np.log10(
        3.0 + event["f_b"] / event["f_s"]
    )
    np.testing.assert_allclose(
        result["brightest_magnitude_ab"], independent_peak_magnitude, atol=1e-12
    )
    source_peak_counts = config["source_electrons_per_exposure"] * 3.0
    blend_counts = (
        config["baseline_electrons_per_exposure"] - config["source_electrons_per_exposure"]
    )
    np.testing.assert_allclose(
        result["representative_peak_pixel_electrons"],
        (source_peak_counts + blend_counts) * config["representative_peak_pixel_fraction"],
        rtol=1e-13,
    )
    assert result["passed"]
    assert not saturation_check(combined * 100, config)["passed"]
