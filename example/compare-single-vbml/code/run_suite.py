"""Run the same physical photometry experiment for three independent noise seeds.

Each seed runs in a separate sequential process. The first seed supplies the
representative figures; inference_suite.json reports paired efficiency ranges.
Additional arguments are passed to inference_speed.py unchanged.
"""

import argparse
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "outputs"

EXPECTED_METHODS = ("table_nuts", "fft_nuts", "vbm_emcee", "table_emcee")
LENS_NAMES = ("t0", "u0", "t_e", "rho")
ARTIFACTS = (
    "inference_data.json",
    "inference_nuts.json",
    "inference_fft_nuts.json",
    "inference_speed.json",
    "inference_event.csv",
    "inference_speed.csv",
    "posterior_singlens_nuts.npz",
    "posterior_fftlog_nuts.npz",
    "posterior_vbm_emcee.npz",
    "posterior_singlens_emcee.npz",
) + tuple(
    f"{name}.{suffix}"
    for name in (
        "inference_speed",
        "inference_recovery",
        "posterior_comparison",
        "trace_comparison",
    )
    for suffix in ("png", "pdf")
)
COMPLETION = "inference_complete.json"


def artifact_hashes(directory):
    result = {}
    for name in ARTIFACTS:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty completed-run artifact: {path}")
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def record_completion(directory, *, started_ns, arguments):
    """Call only after a successful child process; bind its complete artifact set."""
    artifacts = artifact_hashes(directory)
    if any((directory / name).stat().st_mtime_ns < started_ns for name in ARTIFACTS):
        raise ValueError(f"Run retained artifacts older than its start: {directory}")
    temporary = directory / f"{COMPLETION}.tmp"
    temporary.write_text(
        json.dumps({"artifacts": artifacts, "arguments": list(arguments)}, indent=2) + "\n"
    )
    temporary.replace(directory / COMPLETION)


def require_completion(directory):
    marker = directory / COMPLETION
    if not marker.is_file():
        raise ValueError(f"No successful-run completion manifest: {directory}; rerun the suite")
    description = json.loads(marker.read_text())
    if description.get("artifacts") != artifact_hashes(directory):
        raise ValueError(f"Completed-run artifacts changed: {directory}")
    return description


def sampler_settings(row):
    fields = ["backend", "solver", "sampler", "primary_comparison", "series", "ess_parameter_names"]
    if row["sampler"] == "NUTS":
        fields += ["warmup_steps_per_chain", "target_accept_prob"]
    elif row["sampler"] == "emcee":
        fields += ["ensembles", "walkers_per_ensemble", "warmup_steps_per_ensemble"]
    else:
        raise ValueError(f"Unexpected suite sampler: {row['sampler']}")
    return {name: row[name] for name in fields}


def interval(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def summarize(output, seeds):
    completions = [require_completion(output / f"seed-{seed}") for seed in seeds]
    if any(item["arguments"] != completions[0]["arguments"] for item in completions):
        raise ValueError("Suite inference command arguments differ between seeds")
    reports = [
        json.loads((output / f"seed-{seed}" / "inference_speed.json").read_text()) for seed in seeds
    ]
    methods = [row["method"] for row in reports[0]["results"]]
    if tuple(methods) != EXPECTED_METHODS:
        raise ValueError("Suite requires all four expected methods in order")
    truth = reports[0]["injected"]
    if len({report["data_sha256"] for report in reports}) != len(seeds):
        raise ValueError("Suite noise realizations must have distinct data checksums")
    for seed, report in zip(seeds, reports):
        if report["data_seed"] != seed or report["injected"] != truth:
            raise ValueError("Suite requires the same physical event and distinct recorded seeds")
        if report["sampled_parameter_names"] != list(LENS_NAMES):
            raise ValueError("Suite must sample exactly the four lens parameters")
        for key in (
            "priors_transformed",
            "flux_prior",
            "marginalization",
            "sampled_parameter_names",
            "target_ess",
            "reference_data_sha256",
            "reference_provenance",
            "physical_configuration",
            "fit_exposure_order",
            "fit_solvers",
            "benchmark_source_sha256",
            "versions",
            "versions_external",
            "fixed_parameters",
            "source_sha256",
            "python",
            "platform",
            "hardware",
            "devices",
            "device_kind",
            "x64",
            "vbm_interface",
            "vbm_settings",
            "seed",
        ):
            if report[key] != reports[0][key]:
                raise ValueError(f"Suite settings differ between seeds: {key}")
        if [row["method"] for row in report["results"]] != methods:
            raise ValueError("Suite method sets differ between seeds")
        for row, baseline in zip(report["results"], reports[0]["results"]):
            if row["ess_parameter_names"] != list(LENS_NAMES):
                raise ValueError("ESS must use exactly the four lens parameters")
            if sampler_settings(row) != sampler_settings(baseline):
                raise ValueError(f"Suite sampler settings differ: {row['method']}")
        if report["exposure_refinement"]["passed"] is not True:
            raise ValueError(f"Exposure validation failed for seed {seed}")
        if report["vbm_interface_agreement"]["passed"] is not True:
            raise ValueError(f"VBM interface validation failed for seed {seed}")
    by_seed = [{row["method"]: row for row in report["results"]} for report in reports]
    summary = {}
    for method in methods:
        rows = [group[method] for group in by_seed]
        summary[method] = dict(
            diagnostics_pass=all(row["diagnostics_pass"] for row in rows),
            **{
                key: interval([row[key] for row in rows])
                for key in (
                    "total_seconds",
                    "min_ess",
                    "min_ess_per_second",
                    "estimated_seconds_to_target_ess",
                )
            },
            posterior_mean_minus_truth={
                name: interval([row["parameters"][name]["mean"] - value for row in rows])
                for name, value in truth.items()
            },
            truth_in_90pct_interval_count={
                name: sum(
                    row["parameters"][name]["q05"] <= value <= row["parameters"][name]["q95"]
                    for row in rows
                )
                for name, value in truth.items()
            },
        )
    ratios = {
        method: interval(
            [
                group["table_nuts"]["min_ess_per_second"] / group[method]["min_ess_per_second"]
                for group in by_seed
            ]
        )
        for method in ("fft_nuts", "vbm_emcee", "table_emcee")
    }
    return dict(
        data_seeds=seeds,
        representative_seed=seeds[0],
        replicates=len(seeds),
        target_ess=reports[0]["target_ess"],
        nominal_mcse_fraction_of_posterior_sd=reports[0]["nominal_mcse_fraction_of_posterior_sd"],
        ess_target_scope=reports[0]["ess_target_scope"],
        physical_configuration=reports[0]["physical_configuration"],
        flux_prior=reports[0]["flux_prior"],
        marginalization=reports[0]["marginalization"],
        sampled_parameter_names=reports[0]["sampled_parameter_names"],
        reference_data_sha256=reports[0]["reference_data_sha256"],
        reference_provenance=reports[0]["reference_provenance"],
        data_sha256={str(seed): report["data_sha256"] for seed, report in zip(seeds, reports)},
        reports={str(seed): f"seed-{seed}/inference_speed.json" for seed in seeds},
        methods=summary,
        paired_table_ess_per_second_ratio=ratios,
        diagnostics_pass=all(row["diagnostics_pass"] for row in summary.values()),
        scope="Sequential independent processes, same physical mean and priors, independent Gaussian noise realizations. Ranges summarize three fixed seeds; they are not population confidence intervals. Representative plots use the first seed. Sampler/compile/setup timing scope is defined in each per-seed report.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument(
        "--reference-data",
        type=Path,
        help="reuse or create one validated independent mean for every noise seed",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="summarize already completed per-seed runs"
    )
    args, remaining = parser.parse_known_args()
    if len(set(args.data_seeds)) != len(args.data_seeds):
        parser.error("Data seeds must be distinct")
    if any(value == "--data-seed" or value.startswith("--data-seed=") for value in remaining):
        parser.error("Use --data-seeds for this runner")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.reference_data or args.output_dir / "reference_injection.npz"
    if not args.aggregate_only:
        for seed in args.data_seeds:
            directory = args.output_dir / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / COMPLETION).unlink(missing_ok=True)
            started_ns = time.time_ns()
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("inference_speed.py")),
                    "--data-seed",
                    str(seed),
                    "--reference-data",
                    str(reference),
                    "--output-dir",
                    str(directory),
                    *remaining,
                ],
                check=True,
                env=dict(os.environ, JAX_PLATFORMS="cpu", MPLBACKEND="Agg"),
            )
            record_completion(directory, started_ns=started_ns, arguments=remaining)
    report = summarize(args.output_dir, args.data_seeds)
    if not report["diagnostics_pass"]:
        raise RuntimeError("A suite convergence check failed; inspect per-seed reports")
    representative = args.output_dir / f"seed-{args.data_seeds[0]}"
    for name in (*ARTIFACTS, COMPLETION):
        shutil.copy2(representative / name, args.output_dir / name)
    temporary = args.output_dir / "inference_suite.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output_dir / "inference_suite.json")
    print(json.dumps(report["paired_table_ess_per_second_ratio"], indent=2), flush=True)


if __name__ == "__main__":
    main()
