"""Calibrate CPU execution layouts in fresh, sequential processes."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CANDIDATES = (
    "nuts-sequential-2",
    "nuts-parallel-2",
    "nuts-parallel-4",
    "nuts-vectorized-4",
    "nuts-parallel-8",
    "nuts-parallel-16",
) + tuple(f"vbm-{workers}" for workers in (1, 2, 4, 8, 16))


def measure(args):
    from time import perf_counter
    from types import SimpleNamespace

    import jax
    import numpy as np

    from example._common import save_json
    from example._cpu import cpu_configuration, cpu_seconds
    from example.parallax_inference.code import experiment as exp
    from example.parallax_inference.code import posterior as target
    from example.parallax_inference.code.compare import efficiency, run_nuts, source_hashes
    from example.parallax_inference.code.fit import fit_event
    from example.parallax_inference.code.vbm import VBMPosterior

    jax.config.update("jax_enable_x64", True)
    data = exp.synthetic_data(seed=99, reference=args.reference_data)
    started = perf_counter()
    model = exp.make_model("table")
    initial = fit_event(model, data)
    initialization = perf_counter() - started
    if not initial.success:
        raise RuntimeError("Calibration MAP failed")
    result = dict(
        candidate=args.candidate,
        data_seed=99,
        cpu_configuration=cpu_configuration(),
        benchmark_source_sha256=source_hashes(),
        scope="Execution calibration on a separate noise realization; provisional short-run ESS is not the final converged benchmark.",
    )
    if args.candidate.startswith("nuts"):
        _, method, chains = args.candidate.split("-")
        settings = SimpleNamespace(
            target_ess=5000,
            nuts_warmup=500,
            nuts_samples=1000,
            nuts_max_samples=1000,
            nuts_chains=int(chains),
            nuts_chain_method=method,
            seed=18,
        )
        draws, elapsed, row, diagnostics = run_nuts(
            lambda theta: target.negative_log_posterior(theta, model, data), initial, settings
        )
        row.update(
            efficiency(
                draws.transpose(1, 0, 2),
                elapsed,
                initialization,
                diagnostics["warmup_seconds_including_compile"],
            )
        )
        result.update(result=row, sampling_cpu_seconds=diagnostics["cpu_seconds_including_compile"])
    else:
        workers = int(args.candidate.split("-")[1])
        engine = VBMPosterior(data)
        rng = np.random.default_rng(94)
        states = (
            initial.x
            + rng.normal(size=(16, 16, len(target.NAMES))) @ np.linalg.cholesky(initial.hess_inv).T
        )
        expected = engine(states[0])
        started, cpu_started = perf_counter(), cpu_seconds()
        engine.start_workers(workers)
        try:
            np.testing.assert_array_equal(engine(states[0]), expected)
            setup = perf_counter() - started + engine.setup_seconds
            for index in range(3):
                engine(states[index])
            durations = []
            for index in range(128):
                start = perf_counter()
                engine(states[index % len(states)])
                durations.append(perf_counter() - start)
        finally:
            engine.stop_workers()
        result.update(
            workers=workers,
            setup_seconds=setup,
            median_batch_seconds=float(np.median(durations)),
            mean_batch_seconds=float(np.mean(durations)),
            projected_20000_batches_seconds=setup + 20000 * float(np.mean(durations)),
            cpu_seconds=cpu_seconds() - cpu_started + engine.setup_cpu_seconds,
            wall_seconds=perf_counter() - started + engine.setup_seconds,
            raw_batch_seconds=durations,
            batch_scope="16 posterior states per batch, 1199 epochs and 3 exposure nodes per state. Geometry, native magnifications, IPC, and common joint likelihood included. Worker startup is separate; a 20000-batch projection only guides layout choice.",
        )
    save_json(result, args.candidate, args.output_dir)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("benchmark_source_sha256", "raw_batch_seconds", "result")
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate", choices=CANDIDATES)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.candidate:
        measure(args)
        return
    environment = dict(os.environ, JAX_PLATFORMS="cpu", JAX_NUM_CPU_DEVICES="4", MPLBACKEND="Agg")
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    for candidate in CANDIDATES:
        if candidate.startswith("nuts-parallel"):
            environment["JAX_NUM_CPU_DEVICES"] = str(max(4, int(candidate.split("-")[-1])))
        else:
            environment["JAX_NUM_CPU_DEVICES"] = "4"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "example.parallax_inference.code.execution",
                "--candidate",
                candidate,
                "--reference-data",
                str(args.reference_data),
                "--output-dir",
                str(args.output_dir),
            ],
            check=True,
            env=environment,
        )
    results = [json.loads((args.output_dir / f"{name}.json").read_text()) for name in CANDIDATES]
    (args.output_dir / "execution.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
