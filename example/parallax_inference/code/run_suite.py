"""Repeat the same conditional parallax inference comparison for three noise draws."""

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
ARTIFACTS = (
    "comparison.json",
    "data.csv",
    "nuts_diagnostics.json",
    "nuts_chains.npz",
    "emcee_chains.npz",
    "nuts_posterior.npz",
    "emcee_posterior.npz",
) + tuple(
    f"{name}.{suffix}" for name in ("data", "sampling", "corner") for suffix in ("png", "pdf")
)
REPRESENTATIVE_ARTIFACTS = tuple(
    name
    for name in ARTIFACTS
    if name not in {"nuts_chains.npz", "emcee_chains.npz", "nuts_diagnostics.json"}
)


def hashes(directory):
    if any(
        not (directory / name).is_file() or not (directory / name).stat().st_size
        for name in ARTIFACTS
    ):
        raise ValueError(f"Missing or empty completed-run artifact: {directory}")
    return {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in ARTIFACTS}


def interval(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values))


def summarize(output, seeds):
    reports = []
    commands = []
    for seed in seeds:
        path = output / f"seed-{seed}"
        marker = json.loads((path / "complete.json").read_text())
        if marker["artifacts"] != hashes(path):
            raise ValueError(f"Completed artifacts changed: {path}")
        commands.append(marker["arguments"])
        report = json.loads((path / "comparison.json").read_text())
        if not report["diagnostics_pass"] or report["data_seed"] != seed:
            raise ValueError(f"Incomplete or mismatched run: {path}")
        if [row["method"] for row in report["results"]] != ["nuts", "emcee"]:
            raise ValueError("Both samplers must be present in the recorded order")
        if [(row["solver"], row["sampler_library"]) for row in report["results"]] != [
            ("table", "numpyro"),
            ("vbm", "emcee"),
        ]:
            raise ValueError("Suite requires singlens Table/NumPyro NUTS and VBM/emcee")
        if report["sampled_parameters"] != [
            "t0",
            "u0",
            "t_e",
            "rho",
            "pi_e_n",
            "pi_e_e",
            "a1",
            "f_s",
            "f_b",
        ]:
            raise ValueError("Suite requires joint sampling and ESS for all nine parameters")
        reports.append(report)
    if any(command != commands[0] for command in commands):
        raise ValueError("Suite command arguments differ")
    for key in (
        "benchmark_source_sha256",
        "source_sha256",
        "reference_data_sha256",
        "reference_provenance",
        "physical_configuration",
        "priors",
        "flux_prior",
        "lens_parameters",
        "sampled_parameters",
        "flux_treatment",
        "target_ess",
        "sampler_seed",
        "settings",
        "vbm_settings",
        "branch",
        "versions",
        "versions_external",
        "hardware",
        "cpu_configuration",
        "devices",
        "x64",
        "ess_estimator",
    ):
        if any(report[key] != reports[0][key] for report in reports):
            raise ValueError(f"Suite configuration differs: {key}")
    if len({report["data_sha256"] for report in reports}) != len(seeds):
        raise ValueError("Noise realizations must have distinct data hashes")
    for report in reports:
        if report["fit_solvers"]["table"] != reports[0]["fit_solvers"]["table"]:
            raise ValueError("Table settings differ between seeds")
        for name in ("binary", "header", "table", "adapter"):
            if (
                report["fit_solvers"]["vbm"]["interface"]["sha256"][name]
                != reports[0]["fit_solvers"]["vbm"]["interface"]["sha256"][name]
            ):
                raise ValueError("VBM core or interface differs between seeds")
    grouped = [{row["method"]: row for row in report["results"]} for report in reports]
    return dict(
        data_seeds=seeds,
        representative_seed=seeds[0],
        target_ess=reports[0]["target_ess"],
        source_sha256=reports[0]["source_sha256"],
        benchmark_source_sha256=reports[0]["benchmark_source_sha256"],
        physical_configuration=reports[0]["physical_configuration"],
        reference_data_sha256=reports[0]["reference_data_sha256"],
        data_sha256={str(seed): report["data_sha256"] for seed, report in zip(seeds, reports)},
        method_configuration={
            method: {
                key: grouped[0][method][key]
                for key in ("solver", "backend", "sampler", "sampler_library", "label")
            }
            for method in ("nuts", "emcee")
        },
        vbm_settings=reports[0]["vbm_settings"],
        sampled_parameters=reports[0]["sampled_parameters"],
        flux_treatment=reports[0]["flux_treatment"],
        cpu_configuration=reports[0]["cpu_configuration"],
        execution={
            "nuts": {key: grouped[0]["nuts"][key] for key in ("chains", "chain_method")},
            "emcee": {
                key: grouped[0]["emcee"][key]
                for key in ("ensembles", "walkers_per_ensemble", "workers")
            },
        },
        methods={
            method: {
                key: interval([group[method][key] for group in grouped])
                for key in (
                    "total_seconds",
                    "min_ess",
                    "min_ess_per_second",
                    "min_ess_per_production_second",
                    "total_cpu_seconds",
                    "average_cpu_utilization",
                    "min_ess_per_cpu_second",
                )
            }
            for method in ("nuts", "emcee")
        },
        paired_nuts_over_emcee_ess_rate=interval(
            [
                group["nuts"]["min_ess_per_second"] / group["emcee"]["min_ess_per_second"]
                for group in grouped
            ]
        ),
        maximum_posterior_difference_pooled_sd=max(
            value["pooled_sd"]
            for report in reports
            for value in report["posterior_agreement"].values()
        ),
        maximum_posterior_difference_combined_mcse=max(
            value["combined_mcse"]
            for report in reports
            for value in report["posterior_agreement"].values()
        ),
        diagnostics_pass=all(report["diagnostics_pass"] for report in reports),
        scope="Three sequential independent processes with identical physical mean, priors, method configurations, and ESS criteria; independent fixed Gaussian noise seeds. Sampler order alternates. Ranges are sensitivity checks, not population confidence intervals. The two methods are singlens Table/NumPyro NUTS and VBMicrolensing/emcee. Both sample the u0>0 branch; each report includes signed-mode searches. All unthinned nine-parameter chains are retained.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--data-seeds", type=int, nargs="+", default=[43, 44, 45])
    parser.add_argument("--reference-data", type=Path)
    parser.add_argument("--nuts-chains", type=int, default=16)
    parser.add_argument(
        "--nuts-chain-method", choices=("sequential", "parallel", "vectorized"), default="parallel"
    )
    parser.add_argument("--emcee-workers", type=int, default=16)
    parser.add_argument("--aggregate-only", action="store_true")
    args, remaining = parser.parse_known_args()
    remaining = [
        "--nuts-chains",
        str(args.nuts_chains),
        "--nuts-chain-method",
        args.nuts_chain_method,
        "--emcee-workers",
        str(args.emcee_workers),
        *remaining,
    ]
    if len(set(args.data_seeds)) != len(args.data_seeds):
        parser.error("Data seeds must be distinct")
    if any(value.startswith(("--data-seed", "--validate-only")) for value in remaining):
        parser.error("Use --data-seeds and complete sampling runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.reference_data or args.output_dir / "reference_injection.npz"
    environment = dict(
        os.environ,
        JAX_PLATFORMS="cpu",
        MPLBACKEND="Agg",
        JAX_NUM_CPU_DEVICES=str(args.nuts_chains if args.nuts_chain_method == "parallel" else 1),
    )
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    if not args.aggregate_only:
        for seed in args.data_seeds:
            output = args.output_dir / f"seed-{seed}"
            output.mkdir(parents=True, exist_ok=True)
            (output / "complete.json").unlink(missing_ok=True)
            started = time.time_ns()
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "example.parallax_inference.code.compare",
                    "--data-seed",
                    str(seed),
                    "--reference-data",
                    str(reference),
                    "--output-dir",
                    str(output),
                    *remaining,
                ],
                check=True,
                env=environment,
            )
            if any((output / name).stat().st_mtime_ns < started for name in ARTIFACTS):
                raise ValueError("A run retained stale artifacts")
            marker = output / "complete.json.tmp"
            marker.write_text(
                json.dumps(dict(arguments=remaining, artifacts=hashes(output)), indent=2) + "\n"
            )
            marker.replace(output / "complete.json")
    result = summarize(args.output_dir, args.data_seeds)
    for name in REPRESENTATIVE_ARTIFACTS:
        shutil.copy2(args.output_dir / f"seed-{args.data_seeds[0]}" / name, args.output_dir / name)
    summary = args.output_dir / "suite.json.tmp"
    summary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    summary.replace(args.output_dir / "suite.json")
    print(json.dumps(result["paired_nuts_over_emcee_ess_rate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
