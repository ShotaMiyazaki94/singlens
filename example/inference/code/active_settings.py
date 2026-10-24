"""Physical and Roman observing scales for an active K-dwarf + free planet.

The mean spectrum and spot contrast use public CK04 atmospheres and F146
effective area. Spot coverage, rotation, and evolution are scenario choices.
This is an idealized observation, not a detector-level performance forecast.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
SOLAR_RADIUS_CM = 6.957e10
PARSEC_CM = 3.085677581491367e18
PLANCK_ERG_S = 6.62607015e-27
LIGHT_ANGSTROM_S = 2.99792458e18
EARTH_MASS_SOLAR = 3.0034896149156e-6
KAPPA_MAS_PER_SOLAR_MASS = 8.144


def _spectral_inputs():
    area = np.genfromtxt(DATA / "roman_f146_sca01_effective_area.csv", delimiter=",", names=True)
    star = np.genfromtxt(DATA / "ck04_5000_4000_logg4p5_solar.csv", delimiter=",", names=True)
    wave = area["wavelength_micron"] * 1e4
    photo, spot = [
        np.interp(wave, star["wavelength_angstrom"], star[name])
        for name in ("photosphere_surface_flux", "spot_surface_flux")
    ]
    return wave, area["effective_area_m2"] * 1e4, photo, spot


def _count_rate(wave, area, flux):
    return float(np.trapezoid(flux * wave * area / (PLANCK_ERG_S * LIGHT_ANGSTROM_S), wave))


def white_noise(relative_flux, settings=None):
    """Flux error in units of total baseline: photon scaling plus a 1 mmag floor."""
    settings = configuration() if settings is None else settings
    flux = np.asarray(relative_flux)
    return np.sqrt(
        settings["noise_photon_fraction"] ** 2 * flux
        + settings["systematic_floor_flux"] ** 2 * flux**2
    )


def saturation_check(relative_flux, settings=None):
    """Check total noiseless source + spot modulation + lensing + blended light."""
    settings = configuration() if settings is None else settings
    flux = np.asarray(relative_flux)
    if not np.all(np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError("Saturation check requires finite positive total flux")
    peak = float(np.max(flux))
    brightest = float(settings["baseline_mag_ab"] - 2.5 * np.log10(peak))
    pixel = float(settings["baseline_max_pixel_electrons"] * peak)
    return dict(
        brightest_total_mag_ab=brightest,
        total_object_electrons_per_exposure=float(settings["baseline_counts_per_exposure"] * peak),
        representative_peak_pixel_electrons=pixel,
        passed=bool(
            brightest >= settings["saturation_guard_mag"]
            and pixel < settings["saturation_fullwell_electrons"]
        ),
    )


def configuration():
    """Return JSON-safe settings; retain distinctions between assumptions and data."""
    wave, area, photosphere, spot = _spectral_inputs()
    mass, radius, temperature, spot_temperature, distance = 0.8, 0.8, 5000.0, 4000.0, 6.0
    coverage, coverage_rms, extinction, source_fraction = 0.10, 0.02, 1.0, 0.8
    lens_mass, lens_distance, proper_motion, impact_ratio = 0.5, 4.0, 5.0, 0.7
    attenuation = 10 ** (-0.4 * extinction * (wave / 16500.0) ** -2)
    dilution = (radius * SOLAR_RADIUS_CM / (distance * 1000 * PARSEC_CM)) ** 2
    photo_rate = _count_rate(wave, area, photosphere * attenuation * dilution)
    spot_rate = _count_rate(wave, area, spot * attenuation * dilution)
    ratio = spot_rate / photo_rate
    mean_spectrum = (1 - coverage) * photosphere + coverage * spot
    mean_rate = (1 - coverage) * photo_rate + coverage * spot_rate
    zero_rate = _count_rate(wave, area, 3631e-23 * LIGHT_ANGSTROM_S / wave**2)
    source_magnitude = -2.5 * np.log10(mean_rate / zero_rate)
    unextincted_magnitude = -2.5 * np.log10(
        _count_rate(wave, area, mean_spectrum * dilution) / zero_rate
    )
    baseline_magnitude = source_magnitude + 2.5 * np.log10(source_fraction)
    gp_rms = (1 - ratio) * coverage_rms / (1 - coverage * (1 - ratio))
    theta_star = radius * SOLAR_RADIUS_CM / (distance * 1000 * PARSEC_CM) * 206264806.24709636
    pi_rel = 1 / lens_distance - 1 / distance
    theta_e = np.sqrt(KAPPA_MAS_PER_SOLAR_MASS * lens_mass * EARTH_MASS_SOLAR * pi_rel)
    rho = theta_star / theta_e
    u0 = impact_ratio * rho
    effective_temperature = (
        (1 - coverage) * temperature**4 + coverage * spot_temperature**4
    ) ** 0.25
    noise_photon = 0.01 * 10 ** (0.2 * (baseline_magnitude - 21.2))
    floor = np.log(10) / 2.5 * 0.001
    exposure = 66.41188
    base_counts = mean_rate * exposure / source_fraction
    provenance = json.loads((DATA / "active_reference_provenance.json").read_text())
    provenance["settings_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, entry in provenance["inputs"].items():
        if hashlib.sha256((DATA / name).read_bytes()).hexdigest() != entry["extracted_sha256"]:
            raise ValueError(f"Reference input hash mismatch: {name}")
    return dict(
        description="Active main-sequence K source lensed by an isolated sub-Earth-mass planet",
        design_rationale="An active K dwarf supplies observable variability while remaining faint enough for unsaturated Roman exposures. Solar-like variability in ordinary subgiants is much weaker; larger giants require unusually high extinction to remain unsaturated. This source-transiting example requires a separate spatial-spot check because a disk-integrated GP alone does not model differential spot magnification.",
        source_mass_solar=mass,
        source_radius_solar=radius,
        source_photosphere_teff_K=temperature,
        source_teff_K=float(effective_temperature),
        spot_teff_K=spot_temperature,
        source_distance_kpc=distance,
        source_logg_cgs=float(np.log10(27420 * mass / radius**2)),
        atmosphere_logg_cgs=4.5,
        atmosphere_metallicity=0.0,
        source_luminosity_solar=float(radius**2 * (effective_temperature / 5772) ** 4),
        source_theta_mas=float(theta_star),
        temperature_description="5000 K unspotted photosphere and 4000 K spots. source_teff_K is the nominal bolometric mixture temperature if the mean 10% projected coverage also represents surface coverage; it is not a measured effective temperature. The log g = 4.5 CK04 grid approximates the physical gravity.",
        lens_mass_earth=lens_mass,
        lens_mass_solar=float(lens_mass * EARTH_MASS_SOLAR),
        lens_distance_kpc=lens_distance,
        mu_rel_masyr=proper_motion,
        theta_e_mas=float(theta_e),
        pi_rel_mas=float(pi_rel),
        pi_e=float(pi_rel / theta_e),
        t_e_days=float(theta_e / proper_motion * 365.25),
        rho=float(rho),
        u0=float(u0),
        impact_parameter_over_rho=impact_ratio,
        source_mag_ab=float(source_magnitude),
        source_fraction=source_fraction,
        baseline_mag_ab=float(baseline_magnitude),
        unspotted_source_mag_ab=float(-2.5 * np.log10(photo_rate / zero_rate)),
        unextincted_source_mag_ab=float(unextincted_magnitude),
        extinction_at_1p65_micron_mag=extinction,
        extinction_at_2p15_micron_mag=float(extinction * (2.15 / 1.65) ** -2),
        effective_f146_extinction_mag=float(source_magnitude - unextincted_magnitude),
        extinction_description="Illustrative A(lambda) = A(1.65 micron) * (lambda / 1.65 micron)^-2. Monochromatic values are not broad-band H/Ks extinctions.",
        mean_projected_spot_coverage=coverage,
        projected_spot_coverage_rms=coverage_rms,
        spot_to_photosphere_f146_ratio=float(ratio),
        gp_rms=float(gp_rms),
        gp_period_days=5.0,
        gp_coherence_days=20.0,
        variability_description="The fractional F146 RMS is (1-r)*coverage_rms/[1-mean_coverage*(1-r)] using the reddened CK04 spot/photosphere count-rate ratio r. Coverage, 5-day rotation and 20-day exponential coherence are explicit active-star scenario choices, not measurements or a population calibration. The temporal GP describes disk-integrated modulation, not individual resolved spots, flares, or differential spot magnification.",
        cadence_minutes=12.1,
        exposure_seconds=exposure,
        season_days=70.5,
        photon_rate_source=float(mean_rate),
        photon_rate_source_units="detected electrons / second; effective area includes quantum efficiency",
        source_counts_per_exposure=float(mean_rate * exposure),
        baseline_counts_per_exposure=float(base_counts),
        ab_zeropoint_electron_per_second=float(2.5 * np.log10(zero_rate)),
        source_only_poisson_fraction=float(1 / np.sqrt(mean_rate * exposure)),
        baseline_flux_sigma=float(np.hypot(noise_photon, floor)),
        noise_photon_fraction=float(noise_photon),
        systematic_floor_magnitude=0.001,
        systematic_floor_flux=float(floor),
        noise_description="Known Gaussian errors: official survey S/N = 100 at F146_AB = 21.2 scaled as source photon noise, plus an assumed independent 1 mmag per-exposure floor. Photon rates are a separate lower-bound check. This is not a pixel-level ETC or a crowded-field/detector-covariance forecast.",
        noise_anchor_location="Official GBTDS page: High-Cadence Seasons / Observation Specifications / Exposure Specifications, first bullet; GBTDS Overview Table gives F146 exposure time per epoch 66 s and one dither per epoch. The S/N anchor applies to a single exposure, not combined visits.",
        saturation_guard_mag=19.2,
        saturation_fullwell_electrons=100000.0,
        representative_max_pixel_fraction=0.34546,
        baseline_max_pixel_electrons=float(base_counts * 0.34546),
        guard_max_pixel_electrons=float(zero_rate * 10 ** (-0.4 * 19.2) * exposure * 0.34546),
        guard_legacy_saturation_seconds=121.0,
        saturation_description="Check the total instantaneous flux including lensing, variability, and blend. Require magnitude >= 19.2; NASA's 2021 table gives 121 s to saturation at this brightness. Representative SCA01 centered K0V PSF pixel counts provide a separate comparison to the current 100,000 electron/pixel ETC threshold. This excludes later-resultant partial saturation, but is not a full SED/subpixel/detector-specific ETC guarantee.",
        provenance=provenance,
        references={
            "survey": "https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey",
            "read_pattern": "https://roman-docs.stsci.edu/roman-instruments/the-wide-field-instrument/observing-with-the-wfi/wfi-multiaccum-ma-tables/imaging-multiaccum-tables",
            "detector_threshold": "https://roman-docs.stsci.edu/simulation-tools/roman-wfi-exposure-time-calculator/etc-appendix-inputs-and-calculations",
            "psf_and_effective_area": "https://science.nasa.gov/mission/roman-space-telescope/wfi-technical/",
            "legacy_saturation_table": "https://roman.gsfc.nasa.gov/science/apttables2021/performance_tables/2021/run_time_to_saturation_20210614.txt",
            "extinction_law": "https://arxiv.org/abs/0902.3095",
            "young_fgk_rotation": "https://arxiv.org/abs/1606.00057",
            "cool_spot_coverage": "https://arxiv.org/abs/1608.05452",
            "active_k_star_example": "https://arxiv.org/abs/1106.4928",
            "spot_evolution": "https://doi.org/10.1093/mnras/stx1931",
        },
    )


if __name__ == "__main__":
    print(json.dumps(configuration(), indent=2, allow_nan=False))
