"""A published black-hole event and an independent Roman-scale stellar lens."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import singlens as sl


def calculate_parallax_comparison(
    *,
    event,
    theta_e_mas,
    source,
    rho,
    parallax_geometry,
    pi_e_n,
    pi_e_e,
    half_span_day,
    samples=4801,
):
    """Evaluate matching rectilinear and annual-parallax event models."""
    relative_times = jnp.linspace(-half_span_day, half_span_day, samples)
    times = event["t0"] + relative_times  # HJD - 2450000
    geometry = sl.AnnualParallax(**parallax_geometry)
    source_params = {} if source == "point" else dict(rho=rho)
    straight_model = sl.Model(source=source)
    parallax_model = sl.Model(source=source, parallax=geometry)
    straight_params = sl.Parameters(**event, **source_params)
    parallax_params = sl.Parameters(**event, **source_params, pi_e_n=pi_e_n, pi_e_e=pi_e_e)

    magnifications, shifts, trajectories = [], [], []
    for model, params in (
        (straight_model, straight_params),
        (parallax_model, parallax_params),
    ):
        magnifications.append(np.asarray(model.magnification(times, params)))
        shifts.append(np.asarray(theta_e_mas * model.astrometric_shift(times, params)))
        trajectories.append(np.asarray(model.trajectory(times, params)))
    return dict(
        times=np.asarray(relative_times),
        epochs=np.asarray(times),
        event=event,
        rho=rho,
        theta_e_mas=theta_e_mas,
        source=source,
        magnifications=magnifications,
        shifts=shifts,
        trajectories=trajectories,
        parallax_geometry=parallax_geometry,
        pi_e_n=pi_e_n,
        pi_e_e=pi_e_e,
        minimum_u_over_rho=(
            None if rho is None else [float(np.min(np.abs(u)) / rho) for u in trajectories]
        ),
    )


def calculate_long_event():
    """OGLE-2011-BLG-0462 using one published joint geocentric solution."""
    # BEGIN LONG EVENT EXAMPLE
    # Mróz et al. (2022), Table 2: joint fit using Sahu et al. astrometry.
    # All lensing parameters come from the same column of that table.
    t0 = 5763.3286  # HJD - 2450000; geocentric reference epoch is also t0
    t_e = 239.7
    data = calculate_parallax_comparison(
        event=dict(t0=t0, u0=0.00263, t_e=t_e),
        theta_e_mas=5.68,
        source="point",
        rho=None,
        parallax_geometry=dict(
            ra_deg=15 * (17 + 51 / 60 + 40.2082 / 3600),
            dec_deg=-(29 + 53 / 60 + 26.502 / 3600),
            t_ref=t0,
        ),
        pi_e_n=0.073,
        pi_e_e=-0.050,
        half_span_day=3 * t_e,
        samples=24001,
    )
    # END LONG EVENT EXAMPLE
    data.update(
        name="OGLE-2011-BLG-0462 / MOA-2011-BLG-191",
        source_distance_kpc=5.9,
        physical_assumptions=dict(source_distance_kpc=5.9),
        reference=dict(
            url="https://arxiv.org/html/2207.10729v2#S5.T2",
            solution="Mróz et al. (2022), Table 2, joint fit using Sahu et al. astrometry",
            reported_lens_mass_solar=7.88,
            reported_lens_mass_uncertainty_solar=0.82,
            reported_lens_distance_kpc=1.49,
            reported_lens_distance_uncertainty_kpc=0.12,
            source_distance_kpc=5.9,
            source_distance_uncertainty_kpc=1.3,
        ),
        observer_model="Earth annual parallax; intrinsic unblended point-source magnification",
        photometry_note="The movie excludes the large unrelated blend in the observed OGLE photometry",
    )
    return data


def calculate_roman_event():
    """Independent illustrative G-dwarf lens and source, using Earth's orbit."""
    # BEGIN ROMAN EVENT EXAMPLE
    lens_mass_solar = 1.0
    lens_distance_kpc, source_distance_kpc = 6.0, 8.0
    mu_geo_mas_per_year = 5.0
    source_radius_solar = 1.0
    pi_rel_mas = 1 / lens_distance_kpc - 1 / source_distance_kpc
    theta_e_mas = np.sqrt(8.144 * lens_mass_solar * pi_rel_mas)
    t_e = 365.25 * theta_e_mas / mu_geo_mas_per_year
    theta_star_mas = source_radius_solar * (695700000.0 / 149597870700.0) / source_distance_kpc
    rho = theta_star_mas / theta_e_mas
    pi_e = pi_rel_mas / theta_e_mas
    phi_pi = np.deg2rad(30.0)  # north through east
    t0 = 12030.0  # HJD - 2450000; illustrative 2028 season, not a survey schedule
    roman_event = calculate_parallax_comparison(
        event=dict(t0=t0, u0=0.3, t_e=float(t_e)),
        theta_e_mas=float(theta_e_mas),
        source="uniform",
        rho=float(rho),
        parallax_geometry=dict(ra_deg=268.07, dec_deg=-29.23, t_ref=t0),
        pi_e_n=float(pi_e * np.cos(phi_pi)),
        pi_e_e=float(pi_e * np.sin(phi_pi)),
        half_span_day=4 * t_e,
    )
    # Equal intrinsic F146 luminosities and equal foreground extinction.
    blend_to_source_flux = (source_distance_kpc / lens_distance_kpc) ** 2
    roman_event["blend_to_source_flux"] = blend_to_source_flux
    # END ROMAN EVENT EXAMPLE
    roman_event.update(
        name="Illustrative Roman-scale G-dwarf lens and G-dwarf source",
        source_distance_kpc=source_distance_kpc,
        physical_assumptions=dict(
            lens_mass_solar=lens_mass_solar,
            lens_distance_kpc=lens_distance_kpc,
            source_distance_kpc=source_distance_kpc,
            relative_proper_motion_mas_per_year=mu_geo_mas_per_year,
            source_radius_solar=source_radius_solar,
            stellar_types="G-dwarf lens and G-dwarf source",
            flux_band="Roman F146",
            lens_to_source_intrinsic_band_luminosity=1.0,
            lens_minus_source_extinction_mag=0.0,
            parallax_direction_deg_north_through_east=30.0,
        ),
        observer_model="Earth annual-parallax approximation; no Roman L2 orbit, cadence, or noise",
        references=[
            "https://arxiv.org/abs/1808.02490",
            "https://roman-docs.stsci.edu/roman-community-defined-surveys/galactic-bulge-time-domain-survey",
        ],
    )
    assert min(roman_event["minimum_u_over_rho"]) > 50
    return roman_event


def save_long_event_animation(data, output):
    from example.astrometry.code.animation import save_event_animation

    return save_event_animation(
        data["times"],
        data["magnifications"],
        [1000 * shift for shift in data["shifts"]],
        output,
        labels=("No annual parallax", "Annual parallax"),
        colors=("#222222", "#0072B2"),
        line_styles=("--", "-"),
        filename="long_event_animation.gif",
        flux_xticks=np.arange(-600, 601, 300),
        flux_yscale="log",
        origin_label="Origin: unlensed source at each epoch",
    )


def save_blending_animation(data, output):
    from example.astrometry.code.animation import save_event_animation

    return save_event_animation(
        data["times"],
        [data["magnification"], data["normalized_flux"]],
        [1000 * data["source_shift"], 1000 * np.asarray(data["luminous_lens"].relative_shift)],
        output,
        labels=(
            "No blend",
            rf"G-dwarf lens ($F_b/F_s={data['blend_flux'] / data['source_flux']:.2f}$)",
        ),
        colors=("#222222", "#0072B2"),
        line_styles=("--", "-"),
        filename="blending_animation.gif",
        flux_xticks=np.arange(-150, 151, 75),
        flux_ylabel=r"$F/(F_s+F_b)$",
        track_xlabel=r"$\delta\theta_{\mathrm{obs},\tau}$ ($\mu$as)",
        track_ylabel=r"$\delta\theta_{\mathrm{obs},\beta}$ ($\mu$as)",
        origin_label="Origin: unlensed centroid of each model",
    )


def save_long_event_data(data, output):
    columns = [data["times"], data["epochs"]]
    headers = ["time_from_t0_day", "epoch_hjd_minus_2450000"]
    for name, A, shift, trajectory in zip(
        ("no_parallax", "annual_parallax"),
        data["magnifications"],
        data["shifts"],
        data["trajectories"],
    ):
        columns.extend([A, shift.real, shift.imag, trajectory.real, trajectory.imag])
        headers.extend(
            [
                f"A_{name}",
                f"delta_tau_{name}_mas",
                f"delta_beta_{name}_mas",
                f"u_tau_{name}",
                f"u_beta_{name}",
            ]
        )
    np.savetxt(
        Path(output) / "long_event.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
