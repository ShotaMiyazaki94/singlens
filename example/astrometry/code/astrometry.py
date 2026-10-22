"""Finite-source light curves, astrometric shifts, and unresolved blending."""

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import singlens as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from example._common import save_json
from example.astrometry.code.animation import save_source_animation
from example.astrometry.code.long_event import (
    calculate_long_event,
    calculate_roman_event,
    save_blending_animation,
    save_long_event_animation,
    save_long_event_data,
)

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
COLORS = ("#222222", "#0072B2", "#D55E00")
LINE_STYLES = ("--", "-", "-.")
SOURCE_LABELS = ("Point source", "Uniform disk", r"Linear ($a_1=0.6$)")


def publication_style():
    """Style this example locally, leaving the other examples unchanged."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 10,
            "mathtext.fontset": "stix",
            "axes.labelsize": 10,
            "axes.linewidth": 0.7,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "lines.linewidth": 1.25,
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "savefig.facecolor": "white",
        }
    )


# BEGIN BLENDING MODEL
class BlendedObservables(NamedTuple):
    total_flux: jax.Array
    observed_centroid: jax.Array
    baseline_centroid: jax.Array
    relative_shift: jax.Array


def blended_observables(
    magnification,
    source_shift,
    source_position,
    *,
    source_flux,
    blend_flux,
    blend_position,
):
    """Flux and centroid of one unresolved source plus unmicrolensed light.

    Positions and source_shift are complex coordinates in the same frame and
    angular units. Arrays broadcast over epochs. The caller supplies physical
    scalar source_flux > 0 and blend_flux >= 0; no validation is performed.
    The baseline is the same-time unlensed combined-light centroid, not a
    fixed reference-epoch position. Several unmicrolensed blends can be
    represented by their total flux and flux-weighted position at each epoch.
    """
    A = jnp.asarray(magnification)
    delta = jnp.asarray(source_shift)
    theta_s = jnp.asarray(source_position)
    theta_b = jnp.asarray(blend_position)
    g = jnp.asarray(blend_flux) / jnp.asarray(source_flux)
    total_flux = source_flux * (A + g)
    baseline_centroid = (theta_s + g * theta_b) / (1 + g)
    # Compute the small displacement directly, avoiding subtraction of two
    # potentially large absolute centroids.
    relative_shift = A / (A + g) * delta + g / (1 + g) * (A - 1) / (A + g) * (theta_s - theta_b)
    observed_centroid = baseline_centroid + relative_shift
    return BlendedObservables(total_flux, observed_centroid, baseline_centroid, relative_shift)


# END BLENDING MODEL


def calculate_examples():
    """Calculate the first, source-profile comparison only."""
    # BEGIN SOURCE EXAMPLE
    event = dict(t0=0.0, u0=0.1, t_e=0.8)
    times = event["t0"] + event["t_e"] * jnp.linspace(-4.0, 4.0, 4801)  # days
    rho = 0.5
    theta_e_mas = 0.012

    point_model = sl.Model(source="point")
    uniform_model = sl.Model(source="uniform")
    linear_model = sl.Model(source="linear")
    point_params = sl.Parameters(**event)
    uniform_params = sl.Parameters(**event, rho=rho)
    linear_params = sl.Parameters(**event, rho=rho, a1=0.6)

    A_point = point_model.magnification(times, point_params)
    A_uniform = uniform_model.magnification(times, uniform_params)
    A_linear = linear_model.magnification(times, linear_params)
    shift_point_mas = theta_e_mas * point_model.astrometric_shift(times, point_params)
    shift_uniform_mas = theta_e_mas * uniform_model.astrometric_shift(times, uniform_params)
    shift_linear_mas = theta_e_mas * linear_model.astrometric_shift(times, linear_params)
    # Real and imaginary parts are the signed tau and beta components.
    # END SOURCE EXAMPLE

    # The faint complete tracks show an ideal rectilinear continuation, not an
    # annual-parallax prediction over the full wings. No artificial closing
    # segment joins their finite-time endpoints.
    wings = event["t_e"] * np.geomspace(2.0, 1000.0, 1001)
    track_times = jnp.asarray(np.unique(np.concatenate((-wings, times, wings))))
    track_shifts = [
        np.asarray(theta_e_mas * model.astrometric_shift(track_times, params))
        for model, params in (
            (point_model, point_params),
            (uniform_model, uniform_params),
            (linear_model, linear_params),
        )
    ]
    return dict(
        times=np.asarray(times),
        event=event,
        rho=rho,
        theta_e_mas=theta_e_mas,
        a1=float(linear_params.a1),
        magnifications=[np.asarray(A) for A in (A_point, A_uniform, A_linear)],
        shifts=[np.asarray(s) for s in (shift_point_mas, shift_uniform_mas, shift_linear_mas)],
        track_times=np.asarray(track_times),
        track_shifts=track_shifts,
    )


def calculate_blending(roman_event):
    # BEGIN BLENDING EXAMPLE
    magnifications = roman_event["magnifications"]
    shifts = roman_event["shifts"]
    trajectories = roman_event["trajectories"]
    theta_e_mas = roman_event["theta_e_mas"]
    # Positions are relative to the lens and use the same annual-parallax
    # trajectory as the independent Roman-scale stellar event.
    magnification = magnifications[1]
    source_shift = shifts[1]
    source_position_mas = theta_e_mas * trajectories[1]
    source_flux = 1.0
    blend_flux = source_flux * roman_event["blend_to_source_flux"]
    lens_position_mas = 0.0 + 0.0j
    luminous_lens = blended_observables(
        magnification,
        source_shift,
        source_position_mas,
        source_flux=source_flux,
        blend_flux=blend_flux,
        blend_position=lens_position_mas,
    )
    normalized_flux = luminous_lens.total_flux / (source_flux + blend_flux)
    # END BLENDING EXAMPLE
    return dict(
        times=roman_event["times"],
        epochs=roman_event["epochs"],
        event=roman_event["event"],
        rho=roman_event["rho"],
        lensing_event=roman_event,
        magnification=magnification,
        source_shift=source_shift,
        source_position_mas=source_position_mas,
        source_flux=source_flux,
        blend_flux=blend_flux,
        lens_position_mas=lens_position_mas,
        luminous_lens=luminous_lens,
        normalized_flux=np.asarray(normalized_flux),
    )


def verify_observables(data):
    """Check the flux-weighted identity and physical limiting cases."""
    A = np.asarray(data["magnification"])
    delta = np.asarray(data["source_shift"])
    theta_s = np.asarray(data["source_position_mas"])
    fs, fb = data["source_flux"], data["blend_flux"]
    theta_b = data["lens_position_mas"]
    result = data["luminous_lens"]
    direct = (fs * A * (theta_s + delta) + fb * theta_b) / (fs * A + fb)
    np.testing.assert_allclose(result.total_flux, fs * A + fb, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(result.observed_centroid, direct, rtol=2e-13, atol=2e-14)
    np.testing.assert_allclose(
        result.relative_shift, direct - result.baseline_centroid, rtol=2e-11, atol=2e-14
    )
    # A common, time-dependent sky motion changes the absolute centroids but
    # cancels from their same-epoch lensing displacement.
    common_motion = 1.5 + 0.002 * data["times"] + 1j * (-2 + 0.0007 * data["times"])
    translated = blended_observables(
        A,
        delta,
        theta_s + common_motion,
        source_flux=fs,
        blend_flux=fb,
        blend_position=theta_b + common_motion,
    )
    np.testing.assert_allclose(
        translated.relative_shift, result.relative_shift, rtol=2e-13, atol=2e-14
    )
    np.testing.assert_allclose(
        translated.observed_centroid,
        result.observed_centroid + common_motion,
        rtol=2e-13,
        atol=2e-14,
    )
    no_blend = blended_observables(
        A, delta, theta_s, source_flux=fs, blend_flux=0.0, blend_position=theta_b
    )
    np.testing.assert_allclose(no_blend.relative_shift, delta, atol=0, rtol=0)
    unlensed = blended_observables(
        jnp.ones_like(A),
        jnp.zeros_like(delta),
        theta_s,
        source_flux=fs,
        blend_flux=fb,
        blend_position=theta_b,
    )
    np.testing.assert_allclose(unlensed.relative_shift, 0.0, atol=0, rtol=0)
    coincident = blended_observables(
        A, delta, theta_s, source_flux=fs, blend_flux=fb, blend_position=theta_s
    )
    np.testing.assert_allclose(
        coincident.relative_shift, A / (A + fb / fs) * delta, rtol=2e-13, atol=2e-14
    )
    # The recipe is usable inside the same JAX transformations as singlens.
    compiled = jax.jit(blended_observables)(
        A, delta, theta_s, source_flux=fs, blend_flux=fb, blend_position=theta_b
    )
    np.testing.assert_allclose(
        compiled.relative_shift, result.relative_shift, rtol=2e-13, atol=2e-14
    )
    blend_gradient = jax.jacfwd(
        lambda flux: (
            blended_observables(
                A, delta, theta_s, source_flux=fs, blend_flux=flux, blend_position=theta_b
            ).relative_shift
        )
    )(fb)
    assert np.all(np.isfinite(np.asarray(blend_gradient)))


def physical_interpretation(data, *, pi_rel_mas, pi_rel_assumed, motion_frame):
    """One illustrative physical realization, with assumptions kept explicit.

    Angular Einstein radius and annual parallax constrain physical quantities;
    the source distance is an additional assumption. The short rectilinear
    example also needs an assumed relative parallax to assign a lens mass.
    """
    source_distance_kpc = data.get("source_distance_kpc", 8.0)
    kappa_mas_per_solar_mass = 8.144
    solar_to_jupiter_mass = 1047.348644
    au_per_solar_radius = 149597870700.0 / 695700000.0
    theta_e_mas = data["theta_e_mas"]
    lens_mass_solar = theta_e_mas**2 / (kappa_mas_per_solar_mass * pi_rel_mas)
    assumptions = dict(
        data.get("physical_assumptions", {}), source_distance_kpc=source_distance_kpc
    )
    if pi_rel_assumed:
        assumptions["relative_parallax_mas"] = pi_rel_mas
    derived = dict(
        relative_parallax_mas=float(pi_rel_mas),
        lens_distance_kpc=float(1.0 / (1.0 / source_distance_kpc + pi_rel_mas)),
        lens_mass_solar=float(lens_mass_solar),
        lens_mass_jupiter=float(lens_mass_solar * solar_to_jupiter_mass),
        relative_proper_motion_mas_per_year=float(theta_e_mas / data["event"]["t_e"] * 365.25),
    )
    if data.get("rho") is not None:
        theta_star_mas = data["rho"] * theta_e_mas
        derived.update(
            source_angular_radius_mas=float(theta_star_mas),
            source_radius_solar=float(theta_star_mas * source_distance_kpc * au_per_solar_radius),
        )
    return dict(
        assumptions=assumptions,
        relative_parallax_source=(
            "assumed; annual parallax is omitted in this short-event example"
            if pi_rel_assumed
            else "theta_E * hypot(pi_E_N, pi_E_E) from the annual-parallax model inputs"
        ),
        derived=derived,
        proper_motion_frame=motion_frame,
        constants=dict(
            kappa_mas_per_solar_mass=kappa_mas_per_solar_mass,
            solar_to_jupiter_mass=solar_to_jupiter_mass,
            au_per_solar_radius=au_per_solar_radius,
            days_per_year=365.25,
        ),
    )


def event_configuration(data):
    """Self-contained parameters and physical interpretation for one event."""
    config = dict(
        name=data["name"],
        event=data["event"],
        rho=data["rho"],
        source=data["source"],
        theta_e_mas=data["theta_e_mas"],
        parallax_geometry=data["parallax_geometry"],
        pi_e_n=data["pi_e_n"],
        pi_e_e=data["pi_e_e"],
        time_convention="HJD - 2450000; plotted time is t - t0",
        samples=len(data["times"]),
        time_range_day=[float(data["times"][0]), float(data["times"][-1])],
        minimum_u_over_rho=(
            None
            if data["minimum_u_over_rho"] is None
            else dict(zip(("no_parallax", "annual_parallax"), data["minimum_u_over_rho"]))
        ),
        track_vertical_expansion=1.0,
        shift_reference="same-epoch unlensed source center; origin is zero lensing displacement",
        physical_interpretation=physical_interpretation(
            data,
            pi_rel_mas=data["theta_e_mas"] * np.hypot(data["pi_e_n"], data["pi_e_e"]),
            pi_rel_assumed=False,
            motion_frame="geocentric at t_ref",
        ),
    )
    for key in ("reference", "references", "observer_model", "photometry_note"):
        if key in data:
            config[key] = data[key]
    return config


def save_data(source, long_event, blending, output):
    output.mkdir(parents=True, exist_ok=True)
    columns = [source["times"]]
    headers = ["time_day"]
    for name, A, shift in zip(
        ("point", "uniform", "linear"), source["magnifications"], source["shifts"]
    ):
        columns.extend([A, shift.real, shift.imag])
        headers.extend([f"A_{name}", f"delta_tau_{name}_mas", f"delta_beta_{name}_mas"])
    np.savetxt(
        output / "source_profiles.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
    columns = [source["track_times"]]
    headers = ["time_day"]
    for name, shift in zip(("point", "uniform", "linear"), source["track_shifts"]):
        columns.extend([shift.real, shift.imag])
        headers.extend([f"delta_tau_{name}_mas", f"delta_beta_{name}_mas"])
    np.savetxt(
        output / "tracks.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
    save_long_event_data(long_event, output)
    columns = [
        blending["times"],
        blending["epochs"],
        blending["magnification"],
        blending["normalized_flux"],
        blending["source_position_mas"].real,
        blending["source_position_mas"].imag,
        blending["source_shift"].real,
        blending["source_shift"].imag,
    ]
    headers = [
        "time_from_t0_day",
        "epoch_hjd_minus_2450000",
        "A_uniform_annual_parallax",
        "normalized_blended_flux",
        "source_tau_mas",
        "source_beta_mas",
        "source_shift_tau_mas",
        "source_shift_beta_mas",
    ]
    for field in ("observed_centroid", "baseline_centroid", "relative_shift"):
        value = np.asarray(getattr(blending["luminous_lens"], field))
        columns.extend([value.real, value.imag])
        headers.extend([f"luminous_lens_{field}_tau_mas", f"luminous_lens_{field}_beta_mas"])
    np.savetxt(
        output / "blending.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(headers),
        comments="",
        fmt="%.12g",
    )
    save_json(
        dict(
            source_example=dict(
                event=source["event"],
                rho=source["rho"],
                a1=source["a1"],
                theta_e_mas=source["theta_e_mas"],
                samples=len(source["times"]),
                time_range_day=[float(source["times"][0]), float(source["times"][-1])],
                track_samples=len(source["track_times"]),
                track_time_range_day=[
                    float(source["track_times"][0]),
                    float(source["track_times"][-1]),
                ],
                track_max_abs_tau=1000.0,
                track_interpretation="ideal rectilinear continuation, not an annual-parallax forecast",
                track_vertical_expansion=5.0,
                animation_time_range_day=[float(source["times"][0]), float(source["times"][-1])],
                shift_reference="same-epoch unlensed source center; origin is zero lensing displacement",
                physical_interpretation=physical_interpretation(
                    source,
                    pi_rel_mas=0.1,
                    pi_rel_assumed=True,
                    motion_frame="rectilinear short-event approximation",
                ),
            ),
            long_event=dict(event_configuration(long_event), flux_yscale="log"),
            blending=dict(
                event_configuration(blending["lensing_event"]),
                source_flux=blending["source_flux"],
                blend_flux=blending["blend_flux"],
                lens_position_mas=[0.0, 0.0],
                coordinate_frame="tau + 1j*beta; positions relative to the lens",
                shift_reference="same-time unlensed source-plus-lens centroid",
                comparison_reference="each curve uses its own same-epoch unlensed centroid",
                flux_ratio_interpretation="F146: equal intrinsic band luminosity and foreground extinction; g=(D_S/D_L)^2=16/9",
                track_vertical_expansion=1.0,
            ),
            animation=dict(frames=241, frames_per_second=20, duration_ms=50),
            plot_angular_unit="microarcsecond; CSV data use milliarcseconds",
            jax_x64=bool(jax.config.jax_enable_x64),
        ),
        "configuration",
        output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    publication_style()
    source = calculate_examples()
    long_event = calculate_long_event()
    roman_event = calculate_roman_event()
    blending = calculate_blending(roman_event)
    verify_observables(blending)
    save_data(source, long_event, blending, args.output_dir)
    save_source_animation(source, args.output_dir)
    save_long_event_animation(long_event, args.output_dir)
    save_blending_animation(blending, args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
