"""Finite-source batch timings: singlens, VBMicrolensing, MulensModel.

Run with .[examples,validation], or use --jax-only without reference libraries.
Native batch APIs are timed; no Python per-observation reference loop is used.
"""

import argparse
import csv
import importlib.metadata
import sys
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter

import singlens as sl

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from example._common import metadata, save_figure, save_json, style

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"

EVENT = {"t0": 0.0, "u0": 0.005, "t_e": 20.0, "rho": 0.01, "a1": 0.5}
SOURCES = ("uniform", "linear")
EVALUATION_COUNTS = (10, 30, 100, 300, 1000)
FFT_BINS = (512, 1024, 2048, 4096, 8192, 16384)


def evaluators(source, times, *, compare=True, vbm_tolerances=(1e-3,), n_fft=sl.FFTConfig().n_fft):
    """Prepare callables on the same geometry; setup/input preparation is untimed."""
    model = sl.Model(source=source, fft=None if source == "point" else sl.FFTConfig(n_fft=n_fft))
    params = sl.Parameters(
        **{k: EVENT[k] for k in ("t0", "u0", "t_e")},
        rho=None if source == "point" else EVENT["rho"],
        a1=EVENT["a1"] if source == "linear" else None,
    )
    device_times = jnp.asarray(times)
    jax.block_until_ready((device_times, params))
    compiled = jax.jit(model.magnification)
    result = [
        {
            "implementation": "singlens" if source == "point" else f"singlens {n_fft}",
            "n_fft": None if source == "point" else n_fft,
            "method": "PSPL" if source == "point" else "FFTLog + natural cubic",
            "call": lambda: compiled(device_times, params).block_until_ready(),
        }
    ]
    if not compare:
        return result

    import MulensModel as mm
    import VBMicrolensing

    # VBMicrolensing's batch API takes NATURAL logarithms and physical a1.
    batch_times = np.asarray(times).tolist()
    theta = [np.log(EVENT["u0"]), np.log(EVENT["t_e"]), EVENT["t0"]]
    if source != "point":
        theta.append(np.log(EVENT["rho"]))
    for tolerance in (min(vbm_tolerances),) if source == "point" else vbm_tolerances:
        vbm = VBMicrolensing.VBMicrolensing()
        vbm.Tol, vbm.RelTol = tolerance, tolerance
        vbm.a1 = EVENT["a1"] if source == "linear" else 0.0
        vbm.minannuli = 2 if source == "linear" else 1
        batch = vbm.PSPLLightCurve if source == "point" else vbm.ESPLLightCurve
        result.append(
            {
                "implementation": "VBMicrolensing"
                if source == "point"
                else f"VBM Tol={tolerance:g}",
                "method": "PSPLLightCurve" if source == "point" else "ESPLLightCurve",
                "absolute_tolerance": tolerance,
                "relative_tolerance": tolerance,
                "minannuli": vbm.minannuli,
                # Capture each bound method; no late-binding of the loop variable.
                "call": lambda batch=batch: batch(theta, batch_times)[0],
            }
        )
    mm_params = {"t_0": EVENT["t0"], "u_0": EVENT["u0"], "t_E": EVENT["t_e"]}
    if source != "point":
        mm_params["rho"] = EVENT["rho"]
    reference = mm.Model(mm_params)
    method = {
        "point": "point_source",
        "uniform": "finite_source_uniform_Gould94",
        "linear": "finite_source_LD_Yoo04",
    }[source]
    reference.default_magnification_method = method
    gamma = 2 * EVENT["a1"] / (3 - EVENT["a1"]) if source == "linear" else 0.0
    result.append(
        {
            "implementation": "MulensModel",
            "method": method,
            "gamma": gamma,
            "call": lambda: reference.get_magnification(times, gamma=gamma),
        }
    )
    return result


def timed(call, repeats):
    """Synchronized callables; setup is excluded, first call recorded separately."""
    start = perf_counter()
    output = call()
    first = perf_counter() - start
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(output)):
        raise RuntimeError("Non-finite benchmark output")
    for _ in range(3):
        call()
    durations = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        durations.append(perf_counter() - start)
    return dict(
        first_call_seconds=first,
        raw_seconds=durations,
        median_seconds=float(np.median(durations)),
        p10_seconds=float(np.percentile(durations, 10)),
        p90_seconds=float(np.percentile(durations, 90)),
    )


def finite_counts(times):
    """Observation counts inside the source and within three source radii."""
    u = np.hypot((times - EVENT["t0"]) / EVENT["t_e"], EVENT["u0"])
    return dict(
        n_within_source=int(np.sum(u < EVENT["rho"])),
        n_within_three_radii=int(np.sum(u < 3 * EVENT["rho"])),
    )


def measure(
    repeats=100,
    *,
    compare=True,
    epochs=EVALUATION_COUNTS,
    bins=(sl.FFTConfig().n_fft,),
    vbm_tolerances=(1e-3,),
):
    rows = []
    rng = np.random.default_rng(72)
    for source in SOURCES:
        for count in epochs:
            times = np.linspace(-1, 1, count)
            implementations = evaluators(
                source, times, compare=compare, vbm_tolerances=vbm_tolerances, n_fft=bins[0]
            )
            for n_fft in bins[1:]:
                implementations += evaluators(source, times, compare=False, n_fft=n_fft)
            for index in rng.permutation(len(implementations)):
                item = implementations[index]
                timing = timed(item["call"], repeats)
                rows.append(
                    dict(
                        source=source,
                        n_times=len(times),
                        time_range_days=[float(times[0]), float(times[-1])],
                        cadence_minutes=float(np.diff(times).mean() * 1440),
                        **finite_counts(times),
                        **{k: v for k, v in item.items() if k != "call"},
                        **timing,
                    )
                )
            print(f"{source}: {count} finite-source evaluations timed", flush=True)
    return {
        **metadata(),
        "repeats": repeats,
        "warmup_calls": 3,
        "timing_order": "implementation order shuffled with seed 72; sequential repeats",
        "event": EVENT,
        "peak_window_days": [-1, 1],
        "bins": list(bins),
        "vbm_tolerances": list(vbm_tolerances) if compare else [],
        "external_versions": {
            name: importlib.metadata.version(name) for name in ("VBMicrolensing", "MulensModel")
        }
        if compare
        else {},
        "scope": "Native batch light-curve APIs on the same two-day peak window and rectilinear event. Input preparation is excluded; JAX is synchronized. First call includes compilation and initialization. API return conversions remain included. VBM uses its native approximations and minannuli=2 for linear limb darkening. MulensModel uses Gould94/Yoo04 small-source approximations. Numerical accuracy is assessed separately; these are not equal-error speedups.",
        "results": rows,
    }


def runtime_scale(durations, factors=(1, 3)):
    limits = 10.0 ** np.array(
        [np.floor(np.log10(durations.min())), np.ceil(np.log10(durations.max()))]
    )
    ticks = [
        factor * 10.0**power
        for power in range(int(np.log10(limits[0])), int(np.log10(limits[1])) + 1)
        for factor in factors
        if limits[0] <= factor * 10.0**power <= limits[1]
    ]
    return limits, ticks


def plot(report):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained", sharey=True)
    counts = sorted({row["n_times"] for row in report["results"]})
    labels = sorted(
        {row["implementation"] for row in report["results"]},
        key=lambda name: (
            0 if name.startswith("singlens") else 1 if name.startswith("VBM") else 2,
            name,
        ),
    )
    colors = ["#247BA0", "#E07A5F", "#3D8067", "#8765A5", "#C59A32"]
    durations = np.array([row["median_seconds"] for row in report["results"]]) * 1e6
    limits, ticks = runtime_scale(durations)
    for ax, source in zip(axes, SOURCES):
        for index, label in enumerate(labels):
            rows = sorted(
                [
                    row
                    for row in report["results"]
                    if row["source"] == source and row["implementation"] == label
                ],
                key=lambda row: row["n_times"],
            )
            display = label
            if label.startswith("singlens") and len(report["bins"]) == 1:
                display = f"singlens ({report['bins'][0]} bins)"
            elif label.startswith("VBM") and len(report["vbm_tolerances"]) == 1:
                display = f"VBM (tol={report['vbm_tolerances'][0]:g})"
            ax.plot(
                [row["n_times"] for row in rows],
                [row["median_seconds"] * 1e6 for row in rows],
                "o-",
                ms=6,
                lw=1.8,
                color=colors[index % len(colors)],
                label=display,
            )
        ax.set(
            title="Uniform source" if source == "uniform" else "Linear limb darkening (a₁ = 0.5)",
            xscale="log",
            yscale="log",
            xlim=(min(counts) / 1.15, max(counts) * 1.15),
            ylim=limits,
        )
        ax.set_xticks(counts, labels=[str(count) for count in counts])
        ax.set_yticks(ticks, labels=[f"{value:g}" for value in ticks])
        ax.grid(axis="y", which="major", color="0.9", lw=0.7)
    axes[0].set_ylabel("Total time [µs, log scale]")
    fig.supxlabel("Number of finite-source evaluations, N")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside upper center", ncol=3)
    return fig


def plot_runtime_vs_bins(report):
    """Vary FFT resolution at one fixed batch size."""
    counts = {row["n_times"] for row in report["results"]}
    if len(counts) != 1:
        raise ValueError("The FFT runtime comparison requires one fixed batch size.")
    bins = sorted({row["n_fft"] for row in report["results"]})
    durations = np.array([row["median_seconds"] for row in report["results"]]) * 1e6
    limits, ticks = runtime_scale(durations, factors=(1, 2, 3, 5))
    fig, ax = plt.subplots(figsize=(14, 6), layout="constrained")
    for source, color, marker, line in zip(
        SOURCES, ["#247BA0", "#9B1B30"], ["o", "s"], ["-", "--"]
    ):
        rows = sorted(
            [row for row in report["results"] if row["source"] == source],
            key=lambda row: row["n_fft"],
        )
        ax.plot(
            [row["n_fft"] for row in rows],
            [row["median_seconds"] * 1e6 for row in rows],
            color=color,
            marker=marker,
            ls=line,
            ms=6,
            lw=1.8,
            label="Uniform source" if source == "uniform" else "Linear limb darkening (a₁ = 0.5)",
        )
    ax.set_xscale("log", base=2)
    ax.set(
        title=f"{next(iter(counts))} finite-source evaluations per batch",
        yscale="log",
        xlim=(min(bins) / 1.15, max(bins) * 1.15),
        ylim=limits,
        xlabel="FFT bins",
        ylabel="Total time [µs, log scale]",
    )
    ax.set_xticks(
        bins,
        labels=[f"{n}\n(default)" if n == sl.FFTConfig().n_fft else str(n) for n in bins],
    )
    ax.set_yticks(ticks, labels=[f"{value:g}" for value in ticks])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", color="0.9", lw=0.7)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2)
    return fig


def save_timings(report, name, output):
    save_json(report, name, output)
    with (Path(output) / f"{name}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["source", "implementation", "n_times", "n_fft", "repeat", "seconds"])
        for row in report["results"]:
            writer.writerows(
                [row["source"], row["implementation"], row["n_times"], row.get("n_fft"), i, seconds]
                for i, seconds in enumerate(row["raw_seconds"])
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=EVALUATION_COUNTS,
        help="number of finite-source magnifications requested per batch",
    )
    parser.add_argument("--bins", type=int, nargs="+", default=[sl.FFTConfig().n_fft])
    parser.add_argument("--vbm-tolerances", type=float, nargs="+", default=[1e-3])
    parser.add_argument("--jax-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.repeats < 1 or any(n < 2 for n in args.epochs):
        parser.error("repeats must be positive and epochs must be at least 2")
    for n_fft in args.bins:
        sl.FFTConfig(n_fft=n_fft)
    if any(not np.isfinite(t) or t <= 0 for t in args.vbm_tolerances):
        parser.error("VBM tolerances must be finite and positive")
    jax.config.update("jax_enable_x64", True)
    report = measure(
        args.repeats,
        compare=not args.jax_only,
        epochs=sorted(set(args.epochs)),
        bins=sorted(set(args.bins)),
        vbm_tolerances=tuple(args.vbm_tolerances),
    )
    save_timings(report, "benchmark", args.output_dir)
    style()
    save_figure(plot(report), "benchmark", args.output_dir)
    resolution = measure(args.repeats, compare=False, epochs=(100,), bins=FFT_BINS)
    save_timings(resolution, "runtime_vs_bins", args.output_dir)
    save_figure(plot_runtime_vs_bins(resolution), "runtime_vs_bins", args.output_dir)
    plt.close("all")


if __name__ == "__main__":
    main()
