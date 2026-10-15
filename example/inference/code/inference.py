"""Roman-like 1L1S injection–recovery with a jointly inferred temporal GP.

The 72-day, 12-minute W149 sampling and magnitude-error table follow
ShotaMiyazaki94/test_microjax/Roman_recovery. The single-lens transit and
active low-mass source's rotational GP are explicit injection choices.
"""

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares, minimize
from tinygp import GaussianProcess, kernels

import singlens as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from example._common import metadata, save_figure, save_json, style
from example._sampling import sample_nuts

TOPIC = Path(__file__).resolve().parents[1]
OUTPUT = TOPIC / "outputs"
ERROR_TABLE = TOPIC / "data/roman_magnitude_error.csv"
NAMES = (
    "t0",
    "u0",
    "t_e",
    "rho",
    "f_s",
    "f_b",
    "gp_sigma",
    "gp_scale",
    "pi_e_n",
    "pi_e_e",
    "a1",
    "gp_period",
)
LABELS = (
    r"$t_0$ [day]",
    r"$u_0$",
    r"$t_E$ [day]",
    r"$\rho$",
    r"$f_s$",
    r"$f_b$",
    r"$\sigma_{\rm GP}$",
    r"$\tau_{\rm GP}$ [day]",
    r"$\pi_{E,N}$",
    r"$\pi_{E,E}$",
    r"$a_1$",
    r"$P_{\rm rot}$ [day]",
)
# Gaussian priors in transformed coordinates; a1 uses a logit, parallax is untransformed.
PRIOR_LOC = np.array(
    [
        0.0,
        np.log(0.01),
        np.log(15.0),
        np.log(0.01),
        0.0,
        0.0,
        np.log(0.01),
        np.log(20.0),
        0.0,
        0.0,
        0.0,
        np.log(5.0),
    ]
)
PRIOR_SCALE = np.array([0.5, 1.0, 0.6, 0.7, 1.0, 5.0, 1.0, 0.7, 0.5, 0.5, 1.0, 0.7])
MAG_ZEROPOINT = 25.0
PARALLAX = dict(ra_deg=266.0, dec_deg=-29.0, t_ref=0.0, time_origin=2458000.0)


class Injection(NamedTuple):
    times: jax.Array
    flux: jax.Array
    error: jax.Array
    truth: dict
    lens_flux: jax.Array
    gp_signal: jax.Array


def build_gp(times, error, params):
    """Intrinsic source variability: a damped oscillator, solved in O(N).

    gp_period is the damped oscillation period; gp_scale is the exponential
    coherence time. gp_sigma and error are in unmagnified source-flux units.
    """
    omega = jnp.hypot(2 * jnp.pi / params["gp_period"], 1 / params["gp_scale"])
    kernel = kernels.quasisep.SHO(
        omega=omega, quality=0.5 * omega * params["gp_scale"], sigma=params["gp_sigma"]
    )
    return GaussianProcess(kernel, times, diag=error**2)


def gp_log_probability(magnification, params, times, flux, error):
    """F = A(t) * (f_s + g(t)) + f_b; include the change-of-units Jacobian."""
    residual = (flux - params["f_b"]) / magnification - params["f_s"]
    gp = build_gp(times, error / magnification, params)
    return gp.log_probability(residual) - jnp.sum(jnp.log(magnification))


def event_parameters(params):
    names = (*NAMES[:4], "pi_e_n", "pi_e_e", "a1")
    return sl.Parameters(**{name: params[name] for name in names if name in params})


def mean_flux(model, times, params):
    return model.flux(times, event_parameters(params), f_s=params["f_s"], f_b=params["f_b"])


def unpack(theta):
    values = [
        theta[0],
        *jnp.exp(theta[1:5]),
        theta[5],
        *jnp.exp(theta[6:8]),
        *theta[8:10],
        jax.nn.sigmoid(theta[10]),
        jnp.exp(theta[11]),
    ]
    return dict(zip(NAMES, values))


def synthetic_event(seed=42, *, variability=True):
    """One unthinned Roman-like season, with separate GP and white-noise draws."""
    model = sl.Model(source="linear", parallax=sl.AnnualParallax(**PARALLAX))
    f_s = 10 ** (-0.4 * (24.744 - MAG_ZEROPOINT))
    truth = dict(
        t0=0.2,
        u0=0.005,
        t_e=16.235,
        rho=0.01,
        f_s=f_s,
        f_b=f_s * (1 - 0.395) / 0.395,
        gp_sigma=0.01 * f_s,
        gp_scale=20.0,
        pi_e_n=0.2,
        pi_e_e=-0.1,
        a1=0.5,
        gp_period=5.0,
    )
    times = jnp.asarray(np.arange(-36.0, 36.0, 12.0 / 1440.0))
    lens_flux = mean_flux(model, times, truth)
    magnitudes = MAG_ZEROPOINT - 2.5 * np.log10(np.asarray(lens_flux))
    table = np.genfromtxt(ERROR_TABLE, delimiter=",", names=True)
    if magnitudes.min() < table["mag"].min() or magnitudes.max() > table["mag"].max():
        raise ValueError("Injected magnitudes must lie inside the Roman error table.")
    mag_error = np.interp(magnitudes, table["mag"], table["err"])
    error = jnp.asarray(np.log(10) / 2.5 * np.asarray(lens_flux) * mag_error)
    gp_key, noise_key = jax.random.split(jax.random.key(seed))
    magnification = (lens_flux - truth["f_b"]) / truth["f_s"]
    gp_signal = (
        magnification * build_gp(times, jnp.zeros_like(times), truth).sample(gp_key)
        if variability
        else jnp.zeros_like(times)
    )
    flux = lens_flux + gp_signal + error * jax.random.normal(noise_key, times.shape)
    return model, Injection(times, flux, error, truth, lens_flux, gp_signal)


def load_injection(output=OUTPUT):
    """Read the exact saved experiment used by the posterior and documentation."""
    output = Path(output)
    report = json.loads((output / "optimization.json").read_text())
    values = np.genfromtxt(output / "synthetic_event.csv", delimiter=",", names=True)
    model = sl.Model(
        source=report["source"],
        fft=sl.FFTConfig(**report.get("fft", {"n_fft": report["n_fft"]})),
        parallax=sl.AnnualParallax(**report["parallax"]),
    )
    return model, Injection(
        jnp.asarray(values["time_day"]),
        jnp.asarray(values["flux"]),
        jnp.asarray(values["flux_error"]),
        report["injected"],
        jnp.asarray(values["lens_flux"]),
        jnp.asarray(values["gp_injected"]),
    )


def negative_log_posterior(theta, model, times, flux, error):
    params = unpack(theta)
    magnification = model.magnification(times, event_parameters(params))
    loglike = gp_log_probability(magnification, params, times, flux, error)
    return -loglike + 0.5 * jnp.sum(((theta - PRIOR_LOC) / PRIOR_SCALE) ** 2)


def fit_event(model, times, flux, error):
    """Initialize with white-noise least squares, then jointly fit all parameters.

    The final objective is the full GP marginal likelihood plus fixed priors,
    including the log determinant. Neither stage receives the injected truth.
    """

    def residual(theta):
        params = sl.Parameters(
            t0=theta[0],
            u0=jnp.exp(theta[1]),
            t_e=jnp.exp(theta[2]),
            rho=jnp.exp(theta[3]),
            pi_e_n=0.0,
            pi_e_e=0.0,
            a1=0.5,
        )
        return (model.fit_flux(times, params, flux, error).model_flux - flux) / error

    evaluate, jacobian = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    initial = np.array([0.0, np.log(0.008), np.log(14.0), np.log(0.012)])
    preliminary = least_squares(
        lambda x: np.asarray(evaluate(x)),
        initial,
        jac=lambda x: np.asarray(jacobian(x)),
        x_scale="jac",
        max_nfev=200,
    )
    params = sl.Parameters(
        t0=preliminary.x[0],
        u0=np.exp(preliminary.x[1]),
        t_e=np.exp(preliminary.x[2]),
        rho=np.exp(preliminary.x[3]),
        pi_e_n=0.0,
        pi_e_e=0.0,
        a1=0.5,
    )
    fit = model.fit_flux(times, params, flux, error)
    center = np.r_[
        preliminary.x,
        np.log(float(fit.f_s)),
        float(fit.f_b),
        np.log(0.01),
        np.log(20.0),
        0.0,
        0.0,
        0.0,
        np.log(5.0),
    ]
    jac = np.asarray(jax.jit(jax.jacfwd(lambda x: mean_flux(model, times, unpack(x))))(center))
    weighted = jac / np.asarray(error)[:, None]
    covariance = np.linalg.inv(weighted.T @ weighted + np.diag(1 / PRIOR_SCALE**2))
    transform = np.linalg.cholesky(covariance)

    def objective(z):
        return negative_log_posterior(center + transform @ z, model, times, flux, error)

    value_grad = jax.jit(jax.value_and_grad(objective))
    result = minimize(
        lambda z: tuple(np.asarray(v) for v in value_grad(z)),
        np.zeros(len(NAMES)),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-5, "maxiter": 400},
    )
    result.x = center + transform @ result.x
    result.jac = np.linalg.solve(transform.T, result.jac)
    result.hess_inv = transform @ result.hess_inv @ transform.T
    return result, unpack(result.x)


def run_hmc(model, times, flux, error, *, initial, warmup=500, samples=800, seed=18):
    """Two NUTS chains with the full source-variability likelihood and priors."""
    return sample_nuts(
        lambda theta: negative_log_posterior(theta, model, times, flux, error),
        initial,
        unpack,
        warmup=warmup,
        samples=samples,
        seed=seed,
    )


def posterior_magnitudes(model, data, draws, *, seed=23):
    """Pointwise 5/50/95% quantiles of lens + latent GP, without new white noise.

    Pair each joint parameter draw with a conditional GP realization. Keeping
    whole correlated GP paths also avoids independent noise at adjacent times.
    """
    samples = {name: jnp.asarray(draws[name]).reshape(-1) for name in NAMES}
    keys = jax.random.split(jax.random.key(seed), samples["t0"].size)

    @jax.jit
    def evaluate(samples, keys):
        def draw_curve(item):
            params, key = item
            magnification = model.magnification(data.times, event_parameters(params))
            lens = params["f_s"] * magnification + params["f_b"]
            conditional = (
                build_gp(data.times, data.error / magnification, params)
                .condition((data.flux - lens) / magnification, diag=0.0)
                .gp
            )
            return lens + magnification * conditional.sample(key)

        return jax.lax.map(draw_curve, (samples, keys))

    flux = np.asarray(evaluate(samples, keys))
    if not np.isfinite(flux).all() or np.any(flux <= 0):
        raise ValueError("Posterior flux must be finite and positive to plot magnitudes.")
    return np.percentile(MAG_ZEROPOINT - 2.5 * np.log10(flux), [5, 50, 95], axis=0)


def binned_residuals(data, width=0.1):
    """Inverse-variance mean mag residuals and white-noise-only 1-sigma errors.

    Propagate the known flux errors at the noiseless injected lens + GP flux
    to avoid weighting by noisy observed fluxes. The GP covariance is excluded
    from the error bars; binning is for display only.
    """
    times = np.asarray(data.times)
    residual = -2.5 * np.log10(np.asarray(data.flux / data.lens_flux))
    sigma = 2.5 / np.log(10) * np.asarray(data.error / (data.lens_flux + data.gp_signal))
    weights = 1 / sigma**2
    # Round away floating-point drift at exact bin boundaries.
    indices = np.floor(np.round((times - times.min()) / width, 9)).astype(int)
    total = np.bincount(indices, weights=weights)
    occupied = total > 0
    mean_time, mean_residual = (
        np.bincount(indices, weights=weights * values)[occupied] / total[occupied]
        for values in (times, residual)
    )
    return mean_time, mean_residual, 1 / np.sqrt(total[occupied])


def plot_recovery(model, data, params=None, *, draws=None):
    """Magnitudes and data-minus-true-lens residuals at the observation times.

    Posterior draws include event, flux, GP hyperparameter, and conditional GP
    uncertainty. The baseline is the injected lens alone, retaining GP and
    white noise in the data residuals. Without draws, show the joint MAP curve.
    """

    def magnitude(flux):
        return MAG_ZEROPOINT - 2.5 * np.log10(flux)

    if draws is None:
        if params is None:
            raise ValueError("Provide fit parameters or posterior draws.")
        magnification = model.magnification(data.times, event_parameters(params))
        lens = params["f_s"] * magnification + params["f_b"]
        gp = build_gp(data.times, data.error / magnification, params)
        gp_mean = gp.condition((data.flux - lens) / magnification, diag=0.0).gp.loc
        model_mag = magnitude(lens + magnification * gp_mean)
        label = "Model + GP (MAP)"
    else:
        low, model_mag, high = posterior_magnitudes(model, data, draws)
        label = "Model + GP median"

    true_mag = magnitude(data.lens_flux)
    fig, axes = plt.subplots(
        2, 1, figsize=(14, 6), sharex=True, height_ratios=[2, 1], layout="constrained"
    )
    axes[0].plot(
        data.times,
        magnitude(data.flux),
        ".",
        ms=1.5,
        alpha=0.35,
        color="#83919C",
        label="Roman-like data",
    )
    if draws is not None:
        axes[0].fill_between(data.times, low, high, color="#247BA0", alpha=0.25)
    axes[0].plot(data.times, model_mag, lw=1.3, label=label)
    axes[0].plot(
        data.times,
        true_mag,
        color="#E07A5F",
        lw=1.0,
        ls="--",
        label="True lens",
    )
    axes[0].invert_yaxis()
    axes[0].set(
        ylabel="W149 [mag]", title="Roman-like injection–recovery · 72 days, 12-minute cadence"
    )
    axes[0].legend(loc="upper right")
    axes[1].axhline(0.0, color="black", lw=0.8, ls="--")
    axes[1].plot(
        data.times,
        magnitude(data.flux) - true_mag,
        ".",
        ms=1.5,
        alpha=0.2,
        color="#83919C",
    )
    bin_times, bin_residual, bin_error = binned_residuals(data)
    axes[1].errorbar(
        bin_times,
        bin_residual,
        yerr=bin_error,
        fmt=".",
        ms=2.0,
        elinewidth=0.45,
        capsize=0,
        color="#37474F",
        alpha=0.8,
        label="0.1-day bins (1σ)",
    )
    if draws is not None:
        axes[1].fill_between(
            data.times,
            low - true_mag,
            high - true_mag,
            color="#247BA0",
            alpha=0.25,
            label="90% credible interval",
        )
    axes[1].plot(data.times, model_mag - true_mag, color="#247BA0", lw=1.2, label=label, zorder=3)
    axes[1].set(xlim=(-36, 36), xlabel="HJD − 2458000 [day]", ylabel="Residual to true lens [mag]")
    axes[1].legend(loc="upper right", ncol=3, fontsize=9)
    return fig


def plot_corner(draws, truth):
    values = np.column_stack([np.asarray(draws[name]).reshape(-1) for name in NAMES])
    return corner.corner(
        values,
        labels=LABELS,
        truths=[truth[name] for name in NAMES],
        truth_color="#E07A5F",
        color="#247BA0",
        show_titles=True,
        quantiles=[0.05, 0.5, 0.95],
        title_fmt=".4g",
        max_n_ticks=3,
        title_kwargs={"fontsize": 10},
        label_kwargs={"fontsize": 11},
        plot_datapoints=False,
        fill_contours=True,
        levels=(0.68, 0.95),
    )


def plot_trace(draws, truth):
    columns = 4
    rows = (len(NAMES) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 6), sharex=True, layout="constrained")
    for ax, name, label in zip(axes.flat, NAMES, LABELS):
        for chain, values in enumerate(draws[name]):
            ax.plot(values, lw=0.5, alpha=0.75, label=f"Chain {chain + 1}")
        ax.axhline(truth[name], color="black", lw=0.8, ls="--", label="Injected")
        ax.set_ylabel(label)
        ax.ticklabel_format(axis="y", useOffset=False)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="outside upper right",
        ncol=3,
        fontsize=10,
    )
    for i, ax in enumerate(axes.flat):
        if i >= len(NAMES):
            ax.set_visible(False)
        elif i + columns >= len(NAMES):
            ax.set_xlabel("Post-warmup draw")
            ax.tick_params(labelbottom=True)
    fig.suptitle("Joint event + GP posterior traces", x=0.01, ha="left")
    return fig


def recovery_rows(truth, point, draws):
    rows = []
    for name in NAMES:
        low, median, high = np.percentile(draws[name], [5, 50, 95])
        rows.append(
            dict(
                parameter=name,
                injected=float(truth[name]),
                map=float(point[name]),
                q05=float(low),
                median=float(median),
                q95=float(high),
            )
        )
    return rows


def data_hash(data):
    values = np.column_stack([data.times, data.flux, data.error]).astype("<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def experiment_metadata(data):
    return {
        **metadata(),
        "tinygp": __import__("tinygp").__version__,
        "data_seed": 42,
        "data_sha256": data_hash(data),
        "observations": int(data.times.size),
        "cadence_minutes": 12.0,
        "season_days": 72.0,
        "band": "W149 (reference convention)",
        "source_magnitude": 24.744,
        "source_fraction": 0.395,
        "magnitude_zeropoint": MAG_ZEROPOINT,
        "source": "linear",
        "n_fft": sl.FFTConfig().n_fft,
        "fft": asdict(sl.FFTConfig()),
        "parallax": PARALLAX,
        "gp_kernel": "SHO (rotation proxy)",
        "gp_units": {"gp_sigma": "unmagnified source flux", "gp_scale": "day", "gp_period": "day"},
        "stellar_variability": {
            "source_type": "active low-mass dwarf (assumed)",
            "mechanism": "disk-integrated rotational spot modulation",
            "fractional_rms": 0.01,
            "period_days": 5.0,
            "coherence_days": 20.0,
            "observed_flux": "A(t) * (f_s + g(t)) + f_b",
            "reference": "https://doi.org/10.1111/j.1365-2966.2012.21932.x",
            "scope": "Activity and GP coherence are explicit scenario choices, not inferred from one apparent magnitude. J-band observations motivate the variability scale, not a calibrated W149 prediction. A spatial spot-crossing model is not included.",
        },
        "error_table_sha256": hashlib.sha256(ERROR_TABLE.read_bytes()).hexdigest(),
        "injected": data.truth,
        "priors_transformed": {
            "order": list(NAMES),
            "log_parameters": [*NAMES[1:5], *NAMES[6:8], "gp_period"],
            "logit_parameters": ["a1"],
            "location": PRIOR_LOC.tolist(),
            "scale": PRIOR_SCALE.tolist(),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hmc", action="store_true")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    style()
    model, data = synthetic_event()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        args.output_dir / "synthetic_event.csv",
        np.column_stack([data.times, data.flux, data.error, data.lens_flux, data.gp_signal]),
        delimiter=",",
        header="time_day,flux,flux_error,lens_flux,gp_injected",
        comments="",
    )
    print(f"Generated {data.times.size} Roman-like observations with a temporal GP.", flush=True)
    start = perf_counter()
    result, point = fit_event(model, data.times, data.flux, data.error)
    save_json(
        {
            **experiment_metadata(data),
            "optimizer_success": bool(result.success),
            "optimizer_message": str(result.message),
            "iterations": result.nit,
            "negative_log_posterior": float(result.fun),
            "elapsed_seconds_including_compile": perf_counter() - start,
            "parameters": {k: float(v) for k, v in point.items()},
            "theta": result.x.tolist(),
            "covariance": result.hess_inv.tolist(),
        },
        "optimization",
        args.output_dir,
    )
    print(f"Joint MAP: success={result.success}, {result.message}", flush=True)
    if not result.success:
        raise RuntimeError(result.message)
    save_figure(plot_recovery(model, data, point), "recovery", args.output_dir)
    if args.hmc:
        draws, report = run_hmc(
            model,
            data.times,
            data.flux,
            data.error,
            initial=result,
            warmup=args.warmup,
            samples=args.samples,
        )
        save_json({**experiment_metadata(data), **report}, "hmc", args.output_dir)
        np.savez_compressed(args.output_dir / "posterior.npz", **draws)
        rows = recovery_rows(data.truth, point, draws)
        with (args.output_dir / "recovery.csv").open("w") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        save_figure(
            plot_recovery(model, data, draws=draws),
            "recovery",
            args.output_dir,
        )
        save_figure(plot_corner(draws, data.truth), "corner", args.output_dir)
        save_figure(plot_trace(draws, data.truth), "trace", args.output_dir)
        print(
            {k: report[k] for k in ("divergences", "max_r_hat", "min_effective_sample_size")},
            flush=True,
        )
    plt.close("all")


if __name__ == "__main__":
    main()
