"""A long, weak-finite-source event with annual parallax."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import singlens as sl


def calculate_long_event():
    # BEGIN LONG EVENT EXAMPLE
    relative_times = jnp.linspace(-540.0, 540.0, 4801)  # days
    event = dict(t0=8000.0, u0=0.35, t_e=180.0)
    times = event["t0"] + relative_times  # HJD - 2450000
    rho = 0.005
    theta_e_mas = 1.2
    geometry = sl.AnnualParallax(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0)
    straight_model = sl.Model(source="uniform")
    parallax_model = sl.Model(source="uniform", parallax=geometry)
    straight_params = sl.Parameters(**event, rho=rho)
    parallax_params = sl.Parameters(**event, rho=rho, pi_e_n=0.12, pi_e_e=0.08)

    magnifications, shifts, trajectories = [], [], []
    for model, params in (
        (straight_model, straight_params),
        (parallax_model, parallax_params),
    ):
        magnifications.append(np.asarray(model.magnification(times, params)))
        shifts.append(np.asarray(theta_e_mas * model.astrometric_shift(times, params)))
        trajectories.append(np.asarray(model.trajectory(times, params)))
    # Both cases use the same finite source; only annual parallax changes.
    # END LONG EVENT EXAMPLE

    minimum_u_over_rho = [float(np.min(np.abs(u)) / rho) for u in trajectories]
    assert min(minimum_u_over_rho) > 50
    return dict(
        times=np.asarray(relative_times),
        epochs=np.asarray(times),
        event=event,
        rho=rho,
        theta_e_mas=theta_e_mas,
        source="uniform",
        magnifications=magnifications,
        shifts=shifts,
        trajectories=trajectories,
        parallax_geometry=dict(ra_deg=266.0, dec_deg=-29.0, t_ref=8000.0),
        pi_e_n=0.12,
        pi_e_e=0.08,
        minimum_u_over_rho=minimum_u_over_rho,
    )


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
        flux_xticks=np.arange(-540, 541, 180),
    )


def save_blending_animation(data, output):
    from example.astrometry.code.animation import save_event_animation

    return save_event_animation(
        data["times"],
        [data["magnification"], data["normalized_flux"]],
        [1000 * data["source_shift"], 1000 * np.asarray(data["luminous_lens"].relative_shift)],
        output,
        labels=("No blend", r"Luminous lens ($F_b/F_s=0.5$)"),
        colors=("#222222", "#0072B2"),
        line_styles=("--", "-"),
        filename="blending_animation.gif",
        flux_xticks=np.arange(-540, 541, 180),
        flux_ylabel=r"$F/(F_s+F_b)$",
        track_xlabel=r"$\delta\theta_{\mathrm{obs},\tau}$ ($\mu$as)",
        track_ylabel=r"$\delta\theta_{\mathrm{obs},\beta}$ ($\mu$as)",
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
