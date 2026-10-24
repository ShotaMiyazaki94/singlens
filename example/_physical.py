"""Physical injection and explicit observing assumptions for the planet benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from example.inference.code import active_settings as spectral


def planet_configuration():
    """Quiet K dwarf and sub-Earth-mass lens; all angular/time scales are derived.

    The detector model combines measured-reference throughput with an effective
    background/read variance calibrated to the survey's S/N anchor. It does not
    replace the crowded-field ETC, detector ramp fitting, or a Roman ephemeris.
    """
    radius, stellar_mass, temperature = 0.8, 0.8, 5000.0
    source_distance, lens_distance, planet_mass = 6.0, 4.0, 0.5
    proper_motion, impact_ratio, source_fraction = 3.0, 0.7, 0.8
    exposure, extinction = 66.0, 1.0
    wave, area, atmosphere, _ = spectral._spectral_inputs()
    attenuation = 10 ** (-0.4 * extinction * (wave / 16500.0) ** -2)
    dilution = (
        radius * spectral.SOLAR_RADIUS_CM / (source_distance * 1000 * spectral.PARSEC_CM)
    ) ** 2
    rate = spectral._count_rate(wave, area, atmosphere * attenuation * dilution)
    zero_rate = spectral._count_rate(wave, area, 3631e-23 * spectral.LIGHT_ANGSTROM_S / wave**2)
    source_mag = -2.5 * np.log10(rate / zero_rate)
    baseline_mag = source_mag + 2.5 * np.log10(source_fraction)
    baseline_counts = rate * exposure / source_fraction
    anchor_counts = zero_rate * 10 ** (-0.4 * 21.2) * exposure
    background_variance = (anchor_counts / 100.0) ** 2 - anchor_counts
    if background_variance <= 0:
        raise ValueError("The throughput cannot support the adopted survey S/N anchor")
    theta_star = (
        radius
        * spectral.SOLAR_RADIUS_CM
        / (source_distance * 1000 * spectral.PARSEC_CM)
        * 206264806.24709636
    )
    pi_rel = 1 / lens_distance - 1 / source_distance
    mass_solar = planet_mass * spectral.EARTH_MASS_SOLAR
    theta_e = np.sqrt(spectral.KAPPA_MAS_PER_SOLAR_MASS * mass_solar * pi_rel)
    rho = theta_star / theta_e
    t_e = theta_e / proper_motion * 365.25
    provenance = json.loads((spectral.DATA / "active_reference_provenance.json").read_text())
    for name, entry in provenance["inputs"].items():
        if (
            hashlib.sha256((spectral.DATA / name).read_bytes()).hexdigest()
            != entry["extracted_sha256"]
        ):
            raise ValueError(f"Reference input hash mismatch: {name}")
    return dict(
        description="Quiet K dwarf transited by an isolated sub-Earth-mass planet",
        event=dict(
            t0=0.2,
            u0=float(impact_ratio * rho),
            t_e=float(t_e),
            rho=float(rho),
            a1=0.4,
            f_s=source_fraction,
            f_b=1 - source_fraction,
        ),
        source_radius_solar=radius,
        source_mass_solar=stellar_mass,
        source_temperature_K=temperature,
        source_distance_kpc=source_distance,
        source_logg_cgs=float(np.log10(27420 * stellar_mass / radius**2)),
        atmosphere_logg_cgs=4.5,
        atmosphere_metallicity=0.0,
        lens_mass_earth=planet_mass,
        lens_mass_solar=mass_solar,
        lens_distance_kpc=lens_distance,
        mu_geo_masyr=proper_motion,
        theta_star_mas=float(theta_star),
        theta_e_mas=float(theta_e),
        pi_rel_mas=float(pi_rel),
        pi_e=float(pi_rel / theta_e),
        source_crossing_hours=float(rho * t_e * 24),
        transit_hours=float(2 * rho * t_e * 24 * np.sqrt(1 - impact_ratio**2)),
        band="F146 AB",
        source_magnitude_ab=float(source_mag),
        baseline_magnitude_ab=float(baseline_mag),
        source_fraction=source_fraction,
        extinction_at_1p65_micron_mag=extinction,
        extinction_law="A(lambda)=A(1.65 micron)*(lambda/1.65 micron)^-2; illustrative sightline",
        limb_darkening="Fixed linear a1=.4, an illustrative band-averaged profile; not derived from the flux-only CK04 spectrum",
        cadence_minutes=12.1,
        season_days=70.5,
        exposure_seconds=exposure,
        exposure_scope="Top-hat average over the nominal 66-second exposure; detector ramp weighting is not modeled",
        observer="Rectilinear short-event approximation; no sampled annual parallax or mass inference",
        source_electrons_per_exposure=float(rate * exposure),
        baseline_electrons_per_exposure=float(baseline_counts),
        ab_zero_electrons_per_second=float(zero_rate),
        anchor_magnitude_ab=21.2,
        anchor_snr=100.0,
        anchor_scope="Photon+effective-background S/N before adding the assumed independent 1 mmag floor",
        anchor_electrons_per_exposure=float(anchor_counts),
        background_read_variance_electrons2=float(background_variance),
        independent_floor_magnitude=0.001,
        fractional_floor=float(np.log(10) / 2.5 * 0.001),
        noise_scope="Known independent Gaussian errors from source+blend Poisson counts, constant effective background/read variance calibrated to S/N=100 at F146_AB=21.2, and an assumed independent 1 mmag floor. Not a pixel-level or crowded-field instrument forecast.",
        blend_scope="Unrelated unresolved background light with fixed injected source fraction; no luminous planet contribution",
        representative_peak_pixel_fraction=0.34546,
        fullwell_electrons=100000.0,
        conservative_brightness_guard_ab=19.2,
        saturation_scope="Representative centered SCA01/K0V PSF pixel and total-brightness guard; not a full SED/subpixel/MA-ramp ETC calculation",
        inputs=provenance["inputs"],
        references={
            "survey": "https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey",
            "detector": "https://roman-docs.stsci.edu/simulation-tools/roman-wfi-exposure-time-calculator/etc-appendix-inputs-and-calculations",
            "extinction": "https://arxiv.org/abs/0902.3095",
        },
        configuration_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        spectral_helper_sha256=hashlib.sha256(Path(spectral.__file__).read_bytes()).hexdigest(),
    )


def photometric_error(mean_flux, config):
    """Known standard deviation in total-baseline flux units, evaluated at truth."""
    mean_flux = np.asarray(mean_flux, dtype=float)
    if not mean_flux.size or np.any(~np.isfinite(mean_flux)) or np.any(mean_flux <= 0):
        raise ValueError("Expected total flux must be finite and positive")
    counts = config["baseline_electrons_per_exposure"]
    variance = mean_flux * counts + config["background_read_variance_electrons2"]
    return np.sqrt(variance / counts**2 + (config["fractional_floor"] * mean_flux) ** 2)


def saturation_check(mean_flux, config):
    """Check the injected total brightness against the documented approximate guards."""
    mean_flux = np.asarray(mean_flux, dtype=float)
    if not mean_flux.size or np.any(~np.isfinite(mean_flux)) or np.any(mean_flux <= 0):
        raise ValueError("Expected total flux must be finite, nonempty, and positive")
    peak = float(np.max(mean_flux))
    magnitude = config["baseline_magnitude_ab"] - 2.5 * np.log10(peak)
    pixel = (
        config["baseline_electrons_per_exposure"]
        * peak
        * config["representative_peak_pixel_fraction"]
    )
    return dict(
        peak_relative_flux=peak,
        brightest_magnitude_ab=float(magnitude),
        representative_peak_pixel_electrons=float(pixel),
        scope=config["saturation_scope"],
        passed=bool(
            magnitude >= config["conservative_brightness_guard_ab"]
            and pixel < config["fullwell_electrons"]
        ),
    )
