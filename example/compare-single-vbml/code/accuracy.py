"""Compare table, FFTLog, and VBM accuracy against independent source integration.

FFT resolution and MulensModel comparisons are retained as diagnostics.
"""

import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

import singlens as sl
from singlens.lensing import FFTConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from example._common import metadata, model_solver_metadata, save_figure, save_json, style

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"
RATIOS = np.unique(
    np.r_[
        [
            0,
            0.01,
            0.1,
            0.3,
            0.5,
            0.7,
            0.85,
            0.9,
            0.95,
            0.98,
            0.99,
            0.995,
            1,
            1.005,
            1.01,
            1.02,
            1.05,
            1.1,
            1.2,
            1.5,
            2,
            3,
            5,
            10,
            19.99,
            20,
            20.01,
            50,
            1e-6,
            0.00999,
            0.01001,
        ],
        np.linspace(0.98, 1.02, 81),
    ]
)
RADII = (1e-6, 1e-5, 1e-4, 0.001, 0.01, 0.1, 1.0, 10.0)
PRIMARY_FFT_BINS = 4096
PRIMARY_VBM_TOLERANCE = 1e-3
PRIMARY_METHODS = (
    ("singlens FFTLog", "FFTLog", "#247BA0"),
    ("singlens table", "Table", "#3D8067"),
    ("VBMicrolensing", "VBMicrolensing", "#9B1B30"),
)


def integral(u, rho, *, a1=0.0, tolerance=1e-9):
    """Lens-centered polar quadrature; the area element cancels A's pole.

    Radial coordinates are scaled by rho, so the tolerance remains meaningful
    for tiny sources. The uniform radial integral has an analytic primitive.
    This reference uses neither FFTLog nor a microlensing lookup table.
    """
    ratio = abs(u) / rho

    def angular(theta):
        delta = np.sqrt(max(0.0, 1 - (ratio * np.sin(theta)) ** 2))
        center = ratio * np.cos(theta)
        lo, hi = max(0.0, center - delta), center + delta
        # Rationalized primitive difference, avoiding cancellation for u >> rho.
        p, q = rho * lo, rho * hi
        hp, hq = np.hypot(p, 2.0), np.hypot(q, 2.0)
        disk = 0.5 * (hi - lo) * (hq + p * (q + p) / (hq + hp))
        if a1 == 0:
            return disk

        def radial(x):
            mu2 = max(0.0, 1 - (x - center) ** 2 - (ratio * np.sin(theta)) ** 2)
            return np.sqrt(mu2) * ((rho * x) ** 2 + 2) / np.hypot(rho * x, 2.0)

        limb = quad(radial, lo, hi, epsabs=tolerance * 0.01, epsrel=tolerance, limit=150)[0]
        return (1 - a1) * disk + a1 * limb

    upper = np.pi if ratio <= 1 else np.arcsin(1 / ratio)
    # At u=rho the ray exits the source at pi/2, not pi (roundoff-safe).
    if ratio == 1:
        upper = np.pi / 2
    value = quad(angular, 0, upper, epsabs=tolerance * 0.01, epsrel=tolerance, limit=150)[0]
    return 2 * value / (np.pi * rho * (1 - a1 / 3))


def measure(bins):
    from importlib.metadata import version

    import MulensModel as mm
    import VBMicrolensing

    rows, samples, reference_checks = [], [], []
    models = {}
    fft_bins = sorted(set(bins) | {PRIMARY_FFT_BINS})
    for source in ("uniform", "linear"):
        a1 = 0.5 if source == "linear" else 0.0
        references = {}
        for rho in RADII:
            u = RATIOS * rho
            reference = np.array([integral(x, rho, a1=a1) for x in u])
            tighter = np.array([integral(x, rho, a1=a1, tolerance=1e-11) for x in u])
            difference = float(np.max(abs(reference / tighter - 1)))
            if difference > 1e-7:
                raise RuntimeError(f"Unconverged reference: {source}, {rho}, {difference}")
            references[rho] = tighter
            reference_checks.append(dict(source=source, rho=rho, max_relative_change=difference))
        print(f"{source}: independent integrals converged", flush=True)
        table = sl.Model(source=source)
        models[source] = {"table": model_solver_metadata(table), "fft": {}}
        evaluate = jax.jit(table.magnification_at)
        for rho, reference in references.items():
            actual = np.asarray(
                evaluate(jnp.asarray(RATIOS * rho), rho=rho, a1=a1 if source == "linear" else None)
            )
            add_result(
                rows,
                samples,
                source,
                "singlens table",
                rho,
                actual,
                reference,
                solver="table",
                primary=True,
            )
        for n_fft in fft_bins:
            model = sl.Model(
                source=source,
                solver="fft" if n_fft == PRIMARY_FFT_BINS else FFTConfig(n_fft=n_fft),
            )
            models[source]["fft"][str(n_fft)] = model_solver_metadata(model)
            evaluate = jax.jit(model.magnification_at)
            for rho, reference in references.items():
                actual = np.asarray(
                    evaluate(
                        jnp.asarray(RATIOS * rho), rho=rho, a1=a1 if source == "linear" else None
                    )
                )
                add_result(
                    rows,
                    samples,
                    source,
                    "singlens FFTLog",
                    rho,
                    actual,
                    reference,
                    solver="fft",
                    n_fft=n_fft,
                    primary=n_fft == PRIMARY_FFT_BINS,
                )
        # VBM's tolerance controls integration/point-source switching, not the
        # interpolation error of the bundled uniform-source ESPL table.
        for tolerance in (1e-3, 1e-6):
            vbm = VBMicrolensing.VBMicrolensing()
            vbm.Tol, vbm.RelTol, vbm.a1 = tolerance, tolerance, a1
            vbm.minannuli = 2 if source == "linear" else 1
            for rho, reference in references.items():
                actual = np.array([vbm.ESPLMag2(float(r * rho), rho) for r in RATIOS])
                add_result(
                    rows,
                    samples,
                    source,
                    "VBMicrolensing",
                    rho,
                    actual,
                    reference,
                    solver="ESPLMag2",
                    tolerance=tolerance,
                    primary=tolerance == PRIMARY_VBM_TOLERANCE,
                )
        for rho, reference in references.items():
            # u0=0 and tE=1 map these positive times directly to separation u.
            model = mm.Model(dict(t_0=0.0, u_0=0.0, t_E=1.0, rho=rho))
            model.default_magnification_method = (
                "finite_source_LD_Yoo04" if a1 else "finite_source_uniform_Gould94"
            )
            # The table method raises on u=0; record this missing point rather
            # than silently substituting another implementation's center.
            actual = np.full_like(RATIOS, np.nan)
            actual[1:] = model.get_magnification(RATIOS[1:] * rho, gamma=2 * a1 / (3 - a1))
            add_result(
                rows,
                samples,
                source,
                "MulensModel",
                rho,
                actual,
                reference,
                solver=model.default_magnification_method,
            )
        print(f"{source}: table, FFT, and external comparisons finished", flush=True)
    report = dict(
        **metadata(),
        comparison="method_accuracy",
        bins=list(bins),
        radii=list(RADII),
        u_over_rho=RATIOS.tolist(),
        physical_a1=0.5,
        primary_methods=[name for name, _, _ in PRIMARY_METHODS],
        primary_configuration=dict(
            table="bundled", n_fft=PRIMARY_FFT_BINS, vbm_tolerance=PRIMARY_VBM_TOLERANCE
        ),
        model_solvers=models,
        reference="Independent lens-centered source quadrature",
        reference_tolerances=[1e-9, 1e-11],
        reference_checks=reference_checks,
        external_versions={name: version(name) for name in ("VBMicrolensing", "MulensModel")},
        scope="Sampled errors, not a uniform bound. Primary comparison uses the packaged "
        "default table, fixed FFTLog, and VBM at the recorded primary settings. "
        "FFT resolution sweeps, the second VBM tolerance, and MulensModel are diagnostics. "
        "FFTLog uses its small-source approximation for rho<1e-4. "
        "VBM Tol and RelTol are both set to each row's tolerance; minannuli=2 for linear "
        "limb darkening and 1 for uniform sources. Non-finite external "
        "values are counted and excluded from finite-point error summaries.",
        results=rows,
    )
    return report, samples


def add_result(
    rows,
    samples,
    source,
    implementation,
    rho,
    actual,
    reference,
    *,
    solver=None,
    n_fft=None,
    tolerance=None,
    primary=False,
):
    relative = actual / reference - 1
    valid = np.isfinite(relative)
    if implementation.startswith("singlens") and not valid.all():
        raise RuntimeError(f"Non-finite {source}/{implementation}/{rho}")
    rows.append(
        dict(
            source=source,
            implementation=implementation,
            solver=solver,
            primary=primary,
            rho=rho,
            n_fft=n_fft,
            absolute_tolerance=tolerance,
            relative_tolerance=tolerance,
            n_valid=int(valid.sum()),
            n_invalid=int((~valid).sum()),
            max_relative_error=float(np.max(abs(relative[valid]))) if valid.any() else None,
            rms_relative_error=float(np.sqrt(np.mean(relative[valid] ** 2)))
            if valid.any()
            else None,
        )
    )
    samples.extend(
        dict(
            source=source,
            implementation=implementation,
            solver=solver,
            primary=primary,
            rho=rho,
            n_fft=n_fft,
            absolute_tolerance=tolerance,
            relative_tolerance=tolerance,
            u_over_rho=float(r),
            magnification=float(a) if np.isfinite(a) else None,
            reference=float(ref),
            relative_error=float(err) if np.isfinite(err) else None,
        )
        for r, a, ref, err in zip(RATIOS, actual, reference, relative)
    )


def plot(report):
    """One fixed configuration per method, over the full sampled radius range."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained", sharey=True)
    for ax, source in zip(axes, ("uniform", "linear")):
        for implementation, label, color in PRIMARY_METHODS:
            rows = sorted(
                (
                    r
                    for r in report["results"]
                    if r["source"] == source
                    and r["implementation"] == implementation
                    and r["primary"]
                ),
                key=lambda r: r["rho"],
            )
            ax.loglog(
                [r["rho"] for r in rows],
                [
                    max(100 * r["max_relative_error"], 1e-13)
                    if r["max_relative_error"] is not None
                    else np.nan
                    for r in rows
                ],
                "o-",
                color=color,
                label=label,
            )
        ax.set(
            title="Uniform source" if source == "uniform" else "Linear limb darkening (a₁ = 0.5)",
            xlabel="Source radius ρ",
        )
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Max sampled |ΔA| / A [%]")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    return fig


def plot_fft_resolution(report):
    """Developer diagnostic: vary only FFT resolution, independently of the table."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained", sharey=True)
    radii = report["radii"]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(radii)))
    for ax, source in zip(axes, ("uniform", "linear")):
        for rho, color in zip(radii, colors):
            rows = sorted(
                (
                    r
                    for r in report["results"]
                    if r["source"] == source
                    and r["solver"] == "fft"
                    and r["rho"] == rho
                    and r["n_fft"] in report["bins"]
                ),
                key=lambda r: r["n_fft"],
            )
            ax.loglog(
                [r["n_fft"] for r in rows],
                [100 * r["max_relative_error"] for r in rows],
                "o-",
                ms=4,
                color=color,
                label=f"ρ = {rho:g}",
            )
        ax.set(
            title="Uniform source" if source == "uniform" else "Linear limb darkening (a₁ = 0.5)",
            xlabel="FFT bins",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(report["bins"], labels=[str(n) for n in report["bins"]])
        ax.axhline(0.1, color="0.4", lw=0.8, ls="--")
    axes[0].set_ylabel("Max sampled |ΔA| / A [%]")
    fig.legend(
        *axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=4, fontsize=11
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bins",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192, 16384],
        help="FFT resolutions for the separate diagnostic; primary FFTLog remains fixed at 4096",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    bins = sorted(set(args.bins))
    for n in bins:
        FFTConfig(n_fft=n)
    jax.config.update("jax_enable_x64", True)
    report, samples = measure(bins)
    save_json(report, "accuracy", args.output_dir)
    with (args.output_dir / "accuracy.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(samples)
    style()
    save_figure(plot(report), "accuracy", args.output_dir)
    diagnostic = report | {
        "comparison": "FFT resolution diagnostic",
        "results": [r for r in report["results"] if r["solver"] == "fft" and r["n_fft"] in bins],
        "model_solvers": {
            source: {"fft": configurations["fft"]}
            for source, configurations in report["model_solvers"].items()
        },
        "scope": "FFTLog resolution diagnostic on the same independent reference grid. "
        "The FFT backend uses its small-source approximation for rho<1e-4. "
        "Sampled errors are not uniform error bounds.",
    }
    diagnostic.pop("primary_methods")
    diagnostic.pop("primary_configuration")
    save_json(diagnostic, "accuracy_fft_resolution", args.output_dir)
    with (args.output_dir / "accuracy_fft_resolution.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(samples[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(row for row in samples if row["solver"] == "fft" and row["n_fft"] in bins)
    save_figure(plot_fft_resolution(diagnostic), "accuracy_fft_resolution", args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
