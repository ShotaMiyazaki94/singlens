"""Conditional spatial-spot stress test for the finite-source active-star example.

This is a noiseless bias audit, not a fit of a resolved stellar surface or a
validation of every temporal GP realization. The spot map is frozen during
the short source transit; local disk-integrated variability is treated as
known. All caps avoid the lens chord, including the counterexamples.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import singlens as sl
from example.inference.code.active_settings import configuration, white_noise
from singlens.lensing import FFTConfig
from singlens.lensing.concentric import ConcentricDisk

jax.config.update("jax_enable_x64", True)
OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "active_ffp"
CASES = {
    "opposite_2pct": [(0, -0.5, 0.02)],
    "opposite_5pct": [(0, -0.5, 0.05)],
    "opposite_10pct": [(0, -0.5, 0.10)],
    "offcenter_2pct": [(0.5, -0.4, 0.02)],
    "near_chord_2pct": [(0, 0.5, 0.02)],
    "near_chord_offcenter_2pct": [(0.4, 0.5, 0.02)],
    "near_chord_5pct": [(0, 0.4, 0.05)],
    "grazing_2pct": [(0, 0.57, 0.02)],
    "distributed_opposite_10pct": [
        (-0.55, -0.45, 0.025),
        (-0.2, -0.5, 0.025),
        (0.2, -0.5, 0.025),
        (0.55, -0.45, 0.025),
    ],
}


def cap_geometry(x, y, projected_area):
    """Spherical-cap geometry in stellar-radius units, including foreshortening."""
    if x * x + y * y >= 1 or projected_area <= 0:
        raise ValueError("A visible spot center and positive projected area are required")
    center = np.array([x, y, np.sqrt(1 - x * x - y * y)])
    if projected_area >= center[2]:
        raise ValueError("The requested projected cap is too large")
    beta = np.arcsin(np.sqrt(projected_area / center[2]))
    if np.arccos(center[2]) + beta >= np.pi / 2:
        raise ValueError("This smooth audit requires fully visible spherical caps")
    e1 = np.cross(center, [0, 0, 1.0])
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-12 else np.array([1.0, 0, 0])
    return center, beta, e1, np.cross(center, e1)


def cap_quadrature(x, y, projected_area, *, a1, chord_y, n_radial=32, n_azimuth=96):
    """Integrate the smooth spot deficit; the lens singularity stays outside."""
    center, beta, e1, e2 = cap_geometry(x, y, projected_area)
    z, weights = np.polynomial.legendre.leggauss(n_radial)
    cosine = np.cos(beta) + (z + 1) / 2 * (1 - np.cos(beta))
    weights = weights / 2 * (1 - np.cos(beta))
    phi = 2 * np.pi * (np.arange(n_azimuth) + 0.5) / n_azimuth
    vectors = cosine[:, None, None] * center + np.sqrt(1 - cosine[:, None, None] ** 2) * (
        np.cos(phi)[None, :, None] * e1 + np.sin(phi)[None, :, None] * e2
    )
    vectors = vectors.reshape(-1, 3)
    surface_weights = np.repeat(weights, n_azimuth) * 2 * np.pi / n_azimuth
    projected_weights = surface_weights * vectors[:, 2]
    intensity = 1 - a1 * (1 - vectors[:, 2])
    flux_weights = projected_weights * intensity / (np.pi * (1 - a1 / 3))
    ymax = center[1] * np.cos(beta) + np.sqrt(1 - center[1] ** 2) * np.sin(beta)
    clearance = chord_y - ymax
    if clearance <= 0:
        raise ValueError("A cap intersects the lens chord")
    information = dict(
        x=x,
        y=y,
        angular_radius_degrees=float(np.degrees(beta)),
        projected_area=projected_area,
        area_quadrature_error=float(abs(projected_weights.sum() / np.pi - projected_area)),
        intensity_weighted_area=float(flux_weights.sum()),
        minimum_chord_clearance_source_radii=float(clearance),
    )
    return vectors[:, :2], flux_weights, information


class SpotAudit:
    """Profile event geometry and both fluxes against a frozen spotted source."""

    def __init__(self, n_fft=4096):
        self.settings = configuration()
        s = self.settings
        self.rho, self.t_e, self.u0 = s["rho"], s["t_e_days"], s["u0"]
        self.t0, self.a1, self.fs = 0.2, 0.4, s["source_fraction"]
        self.fb = 1 - self.fs
        self.ratio = s["spot_to_photosphere_f146_ratio"]
        self.times = np.arange(
            -s["season_days"] / 2, s["season_days"] / 2, s["cadence_minutes"] / 1440
        )
        self.dense_times = np.linspace(self.t0 - 0.3, self.t0 + 0.3, 1201)
        nodes, weights = np.polynomial.legendre.leggauss(3)
        self.exposure_nodes = nodes * s["exposure_seconds"] / (2 * 86400)
        self.exposure_weights = weights / 2
        self.model = sl.Model(source="linear", solver=FFTConfig(n_fft=n_fft))
        self.q_truth = np.array(
            [0, np.log(self.rho), np.log(self.t_e), np.log(self.u0), np.log(self.fs), self.fb]
        )
        self.predict = jax.jit(self._predict)
        self.jacobian = jax.jit(jax.jacfwd(self._predict))
        self.amplification = jax.jit(self._amplification)

    def physical(self, q):
        return dict(
            t0=self.t0 + self.t_e * q[0],
            rho=jnp.exp(q[1]),
            t_e=jnp.exp(q[2]),
            u0=jnp.exp(q[3]),
            fs=jnp.exp(q[4]),
            fb=q[5],
        )

    def _amplification(self, q, times):
        p = self.physical(q)
        exposure = jnp.asarray(times)[:, None] + self.exposure_nodes
        params = sl.Parameters(t0=p["t0"], u0=p["u0"], t_e=p["t_e"], rho=p["rho"], a1=self.a1)
        value = self.model.magnification(exposure.ravel(), params)
        return value.reshape(exposure.shape) @ self.exposure_weights

    def _predict(self, q):
        p = self.physical(q)
        return p["fs"] * self._amplification(q, self.times) + p["fb"]

    def spotted_amplification(self, times, radial, caps, *, n_radial=32, n_azimuth=96):
        positions, weights, information = [], [], []
        geometries = [cap_geometry(*cap)[:2] for cap in caps]
        for i, (center, radius) in enumerate(geometries):
            for other, other_radius in geometries[i + 1 :]:
                if np.arccos(np.clip(center @ other, -1, 1)) <= radius + other_radius:
                    raise ValueError("The spherical caps overlap")
        for cap in caps:
            xy, weight, info = cap_quadrature(
                *cap, a1=self.a1, chord_y=self.u0 / self.rho, n_radial=n_radial, n_azimuth=n_azimuth
            )
            positions.append(xy)
            weights.append(weight)
            information.append(info)
        xy, weight = np.concatenate(positions), np.concatenate(weights)
        exposure_times = (np.asarray(times)[:, None] + self.exposure_nodes).ravel()
        integrals = []
        for start in range(0, len(exposure_times), 128):
            lens_x = (exposure_times[start : start + 128] - self.t0) / self.t_e
            u = np.sqrt(
                (lens_x[:, None] - self.rho * xy[None, :, 0]) ** 2
                + (self.u0 - self.rho * xy[None, :, 1]) ** 2
            )
            point = (u * u + 2) / (u * np.sqrt(u * u + 4))
            # Explicit quadrature contraction avoids BLAS floating-status warnings
            # observed even for finite zero matrices with NumPy 2.1/Accelerate.
            integrals.extend(np.einsum("ij,j->i", point, weight, optimize=False))
        spot_integral = np.asarray(integrals).reshape(-1, 3) @ self.exposure_weights
        spot_flux = weight.sum()
        deficit = 1 - self.ratio
        spotted = (radial - deficit * spot_integral) / (1 - deficit * spot_flux)
        return spotted, dict(
            spots=information,
            weighted_covering_fraction=float(spot_flux),
            unlensed_flux_deficit=float(deficit * spot_flux),
        )

    def refit(self, flux, sigma):
        def residual(q):
            return (np.asarray(self.predict(q)) - flux) / sigma

        def jacobian(q):
            return np.asarray(self.jacobian(q)) / sigma[:, None]

        low = np.array([-1, np.log(0.1), np.log(0.01), np.log(0.02), np.log(0.05), 0])
        high = np.array([1, np.log(2), np.log(0.5), np.log(2), np.log(1.5), 1])
        fit = least_squares(
            residual,
            self.q_truth,
            jac=jacobian,
            bounds=(low, high),
            x_scale="jac",
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-9,
            max_nfev=300,
        )
        fitted = {key: float(value) for key, value in self.physical(fit.x).items()}
        truth = {key: float(value) for key, value in self.physical(self.q_truth).items()}
        jac = jacobian(self.q_truth)
        covariance = np.linalg.inv(jac.T @ jac)
        errors = dict(
            t0=self.t_e * np.sqrt(covariance[0, 0]),
            rho=self.rho * np.sqrt(covariance[1, 1]),
            t_e=self.t_e * np.sqrt(covariance[2, 2]),
            u0=self.u0 * np.sqrt(covariance[3, 3]),
            fs=self.fs * np.sqrt(covariance[4, 4]),
            fb=np.sqrt(covariance[5, 5]),
        )
        return dict(
            success=bool(fit.success),
            message=fit.message,
            nfev=int(fit.nfev),
            chi2=float(np.sum(fit.fun**2)),
            physical=fitted,
            white_known_GP_fisher_sigma=errors,
            bias_in_fisher_sigma={key: (fitted[key] - truth[key]) / errors[key] for key in errors},
            relative_bias={key: fitted[key] / truth[key] - 1 for key in ("rho", "t_e", "u0", "fs")},
            q=fit.x.tolist(),
            jacobian_condition=float(np.linalg.cond(jac)),
        )

    def numerical_reference(self, radial):
        u = self.rho * np.concatenate(
            [np.linspace(0.7, 0.95, 11), [0.98, 0.995, 1, 1.005, 1.02], np.linspace(1.05, 3, 15)]
        )
        params = sl.Parameters(t0=0, u0=0, t_e=1, rho=self.rho, a1=self.a1)
        references = []
        for n_disk, n_limb in ((96, 48), (192, 96)):
            oracle = ConcentricDisk(n_disk=n_disk, n_limb=n_limb)
            oracle_at = jax.jit(oracle.A)
            references.append(np.asarray(oracle_at(u, params.rho, a1=params.a1)))
        fft = np.asarray(jax.jit(self.model.magnification)(u, params))
        selected = abs(self.times - self.t0) < 0.5
        exposure = (self.times[selected, None] + self.exposure_nodes).ravel()
        params = sl.Parameters(t0=self.t0, u0=self.u0, t_e=self.t_e, rho=self.rho, a1=self.a1)
        sampled = (
            np.asarray(
                oracle_at(
                    jnp.abs(self.model.trajectory(exposure, params)), params.rho, a1=params.a1
                )
            ).reshape(-1, 3)
            @ self.exposure_weights
        )
        difference = self.fs * (radial[selected] - sampled)
        sigma = white_noise(self.fs * sampled + self.fb, self.settings)
        return dict(
            reference="ConcentricDisk; 96/48 and 192/96 orders",
            reference_order_max_difference=float(np.max(abs(references[1] - references[0]))),
            contact_grid_max_relative_error=float(np.max(abs(fft / references[1] - 1))),
            sampled_event_max_flux_error=float(np.max(abs(difference))),
            sampled_event_template_chi2=float(np.sum((difference / sigma) ** 2)),
        )


def spot_figure(audit, report, curves, output):
    """Alternative non-crossing maps, differential magnification, and refit residuals."""
    import matplotlib.pyplot as plt

    from example._common import save_figure
    from example.astrometry.code.astrometry import publication_style

    publication_style()
    fig, axes = plt.subplots(
        1, 3, figsize=(12.8, 4.3), gridspec_kw={"width_ratios": [1, 1.25, 1.25]}
    )
    geometry, original, residual = axes
    angle = np.linspace(0, 2 * np.pi, 256)
    geometry.fill(np.cos(angle), np.sin(angle), color=".96", edgecolor=".3", lw=0.9)
    geometry.axhline(audit.u0 / audit.rho, color=".2", ls="--", lw=0.8, label="Lens chord")
    choices = [
        ("opposite_10pct", "Opposite side: 10%", "#0072B2"),
        ("near_chord_offcenter_2pct", "Nearby: 2%", "#D55E00"),
        ("grazing_2pct", "Grazing: 2%", "#009E73"),
    ]
    time = 24 * (audit.dense_times - audit.t0)
    observed_time = 24 * (audit.times - audit.t0)
    radial = curves["dense_radial_flux"]
    sigma = 2500 / np.log(10) * white_noise(radial, audit.settings) / radial
    residual.fill_between(time, -sigma, sigma, color=".85", alpha=0.6, label="Single-exposure 1σ")
    for name, label, color in choices:
        for cap in CASES[name]:
            center, radius, e1, e2 = cap_geometry(*cap)
            boundary = np.cos(radius) * center + np.sin(radius) * (
                np.cos(angle)[:, None] * e1 + np.sin(angle)[:, None] * e2
            )
            geometry.fill(boundary[:, 0], boundary[:, 1], color=color, alpha=0.3, edgecolor=color)
        dense = curves[name + "_dense_flux"]
        fitted = report["cases"][name]["refit"]
        p = fitted["physical"]
        model = (
            p["fs"] * np.asarray(audit.amplification(np.array(fitted["q"]), audit.dense_times))
            + p["fb"]
        )
        original.plot(time, -2500 * np.log10(dense / radial), color=color, lw=1.2, label=label)
        residual.plot(time, -2500 * np.log10(dense / model), color=color, lw=1.2)
        sampled = -2500 * np.log10(curves[name + "_spatial_flux"] / curves[name + "_refitted_flux"])
        selected = abs(observed_time) < 4
        residual.plot(observed_time[selected], sampled[selected], ".", color=color, ms=3)
    geometry.set(
        xlabel=r"$x/R_\star$",
        ylabel=r"$y/R_\star$",
        title="Alternative spot configurations",
        xlim=(-1.05, 1.05),
        ylim=(-1.05, 1.05),
    )
    geometry.set_aspect("equal")
    geometry.legend(loc="lower center", fontsize=8)
    original.set(title="Before refitting", ylabel="Spatial − scalar model (mmag)")
    residual.set(title="After event + flux refit", ylabel="Spatial − fitted model (mmag)")
    original.legend(loc="lower left", fontsize=7.5)
    residual.legend(loc="lower left", fontsize=7.5)
    for axis in (original, residual):
        axis.set(xlabel=r"Hours from $t_0$", xlim=(-4, 4))
        axis.axhline(0, color=".6", lw=0.6)
    fig.suptitle("Spatial-spot stress test: no lens-chord crossing", fontsize=12, y=0.99)
    fig.text(
        0.08,
        0.015,
        "Frozen caps; known local variability; no noise draw. Dots mark the 12.1-minute observing cadence.",
        fontsize=8,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.87, bottom=0.18, wspace=0.32)
    save_figure(fig, "spatial_spot_check", output)
    plt.close(fig)


def run_spot_check(output=None, *, n_fft=4096, make_figure=False):
    """Return a JSON-safe conditional bias audit; optionally save its products."""
    audit = SpotAudit(n_fft=n_fft)
    radial = np.asarray(audit.amplification(audit.q_truth, audit.times))
    dense_radial = np.asarray(audit.amplification(audit.q_truth, audit.dense_times))
    curves = dict(
        times=audit.times,
        dense_times=audit.dense_times,
        radial_flux=audit.fs * radial + audit.fb,
        dense_radial_flux=audit.fs * dense_radial + audit.fb,
    )
    records = {}
    for name, caps in CASES.items():
        actual, info = audit.spotted_amplification(audit.times, radial, caps)
        dense, _ = audit.spotted_amplification(audit.dense_times, dense_radial, caps)
        refined, _ = audit.spotted_amplification(
            audit.dense_times[::20], dense_radial[::20], caps, n_radial=64, n_azimuth=192
        )
        flux = audit.fs * actual + audit.fb
        approximate = curves["radial_flux"]
        sigma = white_noise(flux, audit.settings)
        fit = audit.refit(flux, sigma)
        info.update(
            template_chi2=float(np.sum(((flux - approximate) / sigma) ** 2)),
            max_observed_anomaly_mmag=float(np.max(abs(-2500 * np.log10(flux / approximate)))),
            quadrature_refinement_max_flux_error=float(
                np.max(abs(audit.fs * (refined - dense[::20])))
            ),
            refit=fit,
        )
        records[name] = info
        curves[name + "_spatial_flux"] = flux
        curves[name + "_refitted_flux"] = np.asarray(audit.predict(np.array(fit["q"])))
        curves[name + "_dense_flux"] = audit.fs * dense + audit.fb
    transit = 2 * audit.rho * audit.t_e * np.sqrt(1 - (audit.u0 / audit.rho) ** 2)
    report = dict(
        n_fft=n_fft,
        settings=audit.settings,
        truth={key: float(value) for key, value in audit.physical(audit.q_truth).items()},
        a1=audit.a1,
        spot_to_photosphere_ratio=audit.ratio,
        observations=len(audit.times),
        source_transit_duration_hours=float(24 * transit),
        rotation_during_transit_degrees=float(360 * transit / audit.settings["gp_period_days"]),
        assumptions=[
            "No noise draw: refits isolate systematic bias, not a detection or posterior.",
            "Known locally constant disk-integrated variability and white errors. Fisher errors condition on the GP; they are not the joint GP-posterior errors.",
            "Frozen spot map during the short transit; rotational motion is quantified separately, not modeled here.",
            "Photosphere and spots share a1=0.4. The F146 CK04 contrast is a band-integrated ratio, not an angle-dependent atmosphere model.",
            "All spherical caps are fully visible, non-overlapping, and strictly separated from the lens chord.",
            "Source fraction and baseline flux refer to the normalized spotted source. Small-cap cases isolate a localized component, not the full 10% mean coverage.",
            "A 2-percentage-point temporal coverage RMS does not determine the spatial spot map or certify a quiet chord.",
            "Successful far-side cases support only that stated geometry and precision; nearby non-crossing counterexamples prevent a blanket scalar-GP claim.",
        ],
        radial_numerical_check=audit.numerical_reference(radial),
        cases=records,
        references=[
            "https://arxiv.org/abs/astro-ph/9906024",
            "https://doi.org/10.1046/j.1365-8711.2002.05496.x",
            "https://arxiv.org/abs/1608.05452",
        ],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "spatial_spot_check.json").write_text(json.dumps(report, indent=2) + "\n")
        np.savez_compressed(output / "spatial_spot_check.npz", **curves)
        if make_figure:
            spot_figure(audit, report, curves, output)
    elif make_figure:
        raise ValueError("An output directory is required to save a figure")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--n-fft", type=int, default=4096)
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args()
    report = run_spot_check(args.output, n_fft=args.n_fft, make_figure=not args.no_figure)
    for name, case in report["cases"].items():
        fit = case["refit"]
        print(
            f"{name}: residual chi2={fit['chi2']:.3g}; rho bias={fit['bias_in_fisher_sigma']['rho']:.3g} sigma; t0 bias={fit['bias_in_fisher_sigma']['t0']:.3g} sigma"
        )


if __name__ == "__main__":
    main()
