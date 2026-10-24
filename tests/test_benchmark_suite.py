"""Prevent accidental publication of mixed or partially completed benchmark runs."""

import json
import os
import subprocess
import sys
from importlib import import_module

import pytest


@pytest.fixture
def completed_suite(tmp_path):
    suite = import_module("example.compare-single-vbml.code.run_suite")
    truth = dict(t0=0.2, u0=0.43, t_e=0.123, rho=0.614, f_s=0.8, f_b=0.2)
    for index, seed in enumerate((43, 44, 45)):
        directory = tmp_path / f"seed-{seed}"
        directory.mkdir()
        for name in suite.ARTIFACTS:
            (directory / name).write_text(f"synthetic artifact for seed {seed}: {name}\n")
        results = []
        for method, factor in zip(suite.EXPECTED_METHODS, (1, 2, 4, 2.5)):
            nuts, vbm = method.endswith("nuts"), method.startswith("vbm")
            seconds = (10 + index) * factor
            row = dict(
                method=method,
                backend="VBM" if vbm else "singlens",
                solver="VBM" if vbm else method.split("_")[0],
                sampler="NUTS" if nuts else "emcee",
                primary_comparison=method != "table_emcee",
                series=2 if nuts else 64,
                total_seconds=seconds,
                min_ess=1000,
                min_ess_per_second=1000 / seconds,
                estimated_seconds_to_target_ess=seconds,
                diagnostics_pass=True,
                ess_parameter_names=["t0", "u0", "t_e", "rho"],
                parameters={
                    name: {"mean": value + 0.01 * index, "q05": value - 0.1, "q95": value + 0.1}
                    for name, value in truth.items()
                },
            )
            if nuts:
                row.update(warmup_steps_per_chain=500, target_accept_prob=0.95)
            else:
                row.update(ensembles=2, walkers_per_ensemble=32, warmup_steps_per_ensemble=1000)
            results.append(row)
        report = dict(
            data_seed=seed,
            data_sha256=f"distinct-noise-{seed}",
            injected=truth,
            priors_transformed={"location": [0] * 4},
            sampled_parameter_names=["t0", "u0", "t_e", "rho"],
            flux_prior={
                "order": ["f_s", "f_b"],
                "mean": [0.8, 0.2],
                "covariance": [[0.0272, -0.0252], [-0.0252, 0.0257]],
                "support": "untruncated Gaussian",
            },
            marginalization="Normalized analytic Gaussian flux integral",
            target_ess=1000,
            reference_data_sha256="common-mean-and-errors",
            reference_provenance={"source": "independent"},
            physical_configuration={"lens_mass_earth": 0.5},
            fit_exposure_order=3,
            fit_solvers={"table": "same-table", "fft": "same-fft"},
            benchmark_source_sha256={"run_suite.py": "measured-source"},
            versions={"jax": "recorded-version"},
            versions_external={"VBMicrolensing": "5.x"},
            fixed_parameters={"a1": 0.4},
            source_sha256={"model.py": "same-source"},
            python="3.12",
            platform="test-platform",
            hardware={"chip": "test-chip"},
            devices=["cpu:0"],
            device_kind="cpu",
            x64=True,
            vbm_interface="array",
            vbm_settings={"Tol": 1e-3, "RelTol": 1e-3, "minannuli": 2},
            seed=18,
            exposure_refinement={"passed": True},
            vbm_interface_agreement={"passed": True},
            nominal_mcse_fraction_of_posterior_sd=1000**-0.5,
            ess_target_scope="synthetic test",
            results=results,
        )
        (directory / "inference_speed.json").write_text(json.dumps(report))
        suite.record_completion(directory, started_ns=0, arguments=[])
    return suite, tmp_path


def alter_report(completed_suite, change, *, seed=44):
    suite, output = completed_suite
    directory = output / f"seed-{seed}"
    path = directory / "inference_speed.json"
    report = json.loads(path.read_text())
    change(report)
    path.write_text(json.dumps(report))
    # Deliberately bind a complete synthetic run with mismatched metadata, so
    # cross-run checks are exercised independently of checksum rejection.
    suite.record_completion(directory, started_ns=0, arguments=[])


def test_valid_completed_suite_aggregates_paired_results(completed_suite):
    suite, output = completed_suite
    report = suite.summarize(output, [43, 44, 45])
    assert report["diagnostics_pass"]
    assert report["representative_seed"] == 43
    assert report["methods"]["table_nuts"]["total_seconds"] == {"min": 10, "median": 11, "max": 12}
    assert report["paired_table_ess_per_second_ratio"]["fft_nuts"] == {
        "min": 2,
        "median": 2,
        "max": 2,
    }
    assert all(
        count == 3
        for count in report["methods"]["table_nuts"]["truth_in_90pct_interval_count"].values()
    )


def test_missing_completion_marker_blocks_aggregation(completed_suite):
    suite, output = completed_suite
    (output / "seed-44" / suite.COMPLETION).unlink()
    with pytest.raises(ValueError, match="completion manifest"):
        suite.summarize(output, [43, 44, 45])


@pytest.mark.parametrize("change", ["missing", "empty", "mutated"])
def test_missing_or_changed_artifact_blocks_completed_run(completed_suite, change):
    suite, output = completed_suite
    path = output / "seed-44" / "inference_recovery.png"
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"" if change == "empty" else b"different figure")
    with pytest.raises(ValueError, match="artifact"):
        suite.summarize(output, [43, 44, 45])


def test_old_artifact_cannot_be_certified_as_a_new_run(completed_suite):
    suite, output = completed_suite
    directory = output / "seed-43"
    marker = directory / suite.COMPLETION
    marker.unlink()
    os.utime(directory / "inference_recovery.png", ns=(1, 1))
    with pytest.raises(ValueError, match="older than its start"):
        suite.record_completion(directory, started_ns=2, arguments=[])
    assert not marker.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("hardware", {"chip": "different-chip"}),
        ("device_kind", "gpu"),
        ("devices", ["cpu:1"]),
        ("x64", False),
        ("python", "3.13"),
        ("vbm_interface", "python"),
        ("vbm_settings", {"Tol": 1e-6}),
        ("seed", 19),
        ("reference_data_sha256", "different-mean"),
        ("flux_prior", {"mean": [0.8, 0.2], "covariance": [[0.1, 0.0], [0.0, 0.1]]}),
        ("marginalization", "Profile likelihood at fitted fluxes"),
    ],
)
def test_mixed_run_settings_block_aggregation(completed_suite, field, value):
    suite, output = completed_suite
    alter_report(completed_suite, lambda report: report.update({field: value}))
    with pytest.raises(ValueError, match="settings differ"):
        suite.summarize(output, [43, 44, 45])


@pytest.mark.parametrize(
    "method,field,value",
    [
        ("table_nuts", "warmup_steps_per_chain", 100),
        ("fft_nuts", "target_accept_prob", 0.8),
        ("vbm_emcee", "walkers_per_ensemble", 48),
        ("table_emcee", "warmup_steps_per_ensemble", 500),
    ],
)
def test_mixed_method_sampler_settings_block_aggregation(completed_suite, method, field, value):
    suite, output = completed_suite

    def change(report):
        next(row for row in report["results"] if row["method"] == method)[field] = value

    alter_report(completed_suite, change)
    with pytest.raises(ValueError, match="sampler settings differ"):
        suite.summarize(output, [43, 44, 45])


def test_unrecorded_cli_setting_changes_are_not_silently_mixed(completed_suite):
    suite, output = completed_suite
    suite.record_completion(output / "seed-44", started_ns=0, arguments=["--nuts-samples", "1200"])
    with pytest.raises(ValueError, match="command arguments differ"):
        suite.summarize(output, [43, 44, 45])


def test_partial_method_list_cannot_be_aggregated(completed_suite):
    suite, output = completed_suite
    alter_report(completed_suite, lambda report: report["results"].pop(), seed=43)
    with pytest.raises(ValueError, match="four expected methods"):
        suite.summarize(output, [43, 44, 45])


@pytest.mark.parametrize("all_seeds", [False, True])
def test_conditional_fluxes_cannot_enter_published_ess(completed_suite, all_seeds):
    suite, output = completed_suite

    def wrong_scope(report):
        report["results"][0]["ess_parameter_names"] += ["f_s", "f_b"]

    for seed in (43, 44, 45) if all_seeds else (44,):
        alter_report(completed_suite, wrong_scope, seed=seed)
    with pytest.raises(ValueError, match="ESS|ess|diagnostic|lens"):
        suite.summarize(output, [43, 44, 45])


def test_wrong_sampled_parameter_scope_is_rejected_even_when_seeds_match(completed_suite):
    suite, output = completed_suite
    for seed in (43, 44, 45):
        alter_report(
            completed_suite,
            lambda report: report["sampled_parameter_names"].extend(["f_s", "f_b"]),
            seed=seed,
        )
    with pytest.raises(ValueError, match="sampled|lens|parameter"):
        suite.summarize(output, [43, 44, 45])


@pytest.mark.parametrize("field", ["exposure_refinement", "vbm_interface_agreement"])
def test_failed_numerical_checks_block_aggregation(completed_suite, field):
    suite, output = completed_suite
    alter_report(completed_suite, lambda report: report[field].update(passed=False))
    with pytest.raises(ValueError, match="validation failed"):
        suite.summarize(output, [43, 44, 45])


def test_aggregate_publication_copies_only_expected_artifacts(completed_suite, monkeypatch):
    suite, output = completed_suite
    (output / "seed-43" / "stale_extra.png").write_bytes(b"unrelated previous output")
    monkeypatch.setattr(sys, "argv", ["run_suite", "--aggregate-only", "--output-dir", str(output)])
    suite.main()
    assert not (output / "stale_extra.png").exists()
    for name in (*suite.ARTIFACTS, suite.COMPLETION):
        assert (output / name).read_bytes() == (output / "seed-43" / name).read_bytes()
    assert json.loads((output / "inference_suite.json").read_text())["diagnostics_pass"]


def test_failed_sampler_cannot_promote_outputs(completed_suite, monkeypatch):
    suite, output = completed_suite
    alter_report(
        completed_suite, lambda report: report["results"][0].update(diagnostics_pass=False)
    )
    monkeypatch.setattr(sys, "argv", ["run_suite", "--aggregate-only", "--output-dir", str(output)])
    with pytest.raises(RuntimeError, match="convergence check failed"):
        suite.main()
    assert not (output / "inference_suite.json").exists()
    assert not (output / "inference_recovery.png").exists()


def test_failed_retry_invalidates_previous_completion_marker(completed_suite, monkeypatch):
    suite, output = completed_suite
    marker = output / "seed-43" / suite.COMPLETION

    def fail(command, **_kwargs):
        assert not marker.exists()
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(suite.subprocess, "run", fail)
    monkeypatch.setattr(sys, "argv", ["run_suite", "--output-dir", str(output)])
    with pytest.raises(subprocess.CalledProcessError):
        suite.main()
    with pytest.raises(ValueError, match="completion manifest"):
        suite.summarize(output, [43, 44, 45])
