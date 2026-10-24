"""Run complete CI shards or focused compatibility checks.

Use --list to inspect the file partition without importing pytest or JAX.
New and nested test files automatically enter the general shard.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUMERICAL_SHARDS = {
    "photometry": {
        "tests/test_fspl.py",
        "tests/test_concentric.py",
        "tests/test_fftlog.py",
        "tests/test_special.py",
        "tests/test_interpolate.py",
    },
    "astrometry_tables": {
        "tests/test_astrometry.py",
        "tests/test_concentric_astrometry.py",
        "tests/test_basis_table.py",
        "tests/test_joint_basis_table.py",
        "tests/test_table_interpolation.py",
        "tests/test_bundled_basis_table.py",
        "tests/test_joint_model.py",
        "tests/test_reference_libraries.py",
    },
}
COMPATIBILITY_FILES = {
    "tests/test_api_contract.py",
    "tests/test_point_source.py",
    "tests/test_model.py",
    "tests/test_magnification.py",
    "tests/test_astrometry_api.py",
    "tests/test_parallax.py",
    "tests/test_upstream.py",
    "tests/test_likelihood.py",
    "tests/test_fftlog.py",
    "tests/test_special.py",
    "tests/test_interpolate.py",
    "tests/test_table_interpolation.py",
    "tests/test_active_spots.py",
    "tests/test_example_cpu.py",
    "tests/test_fetch_example_data.py",
}
COMPATIBILITY_CASES = {
    "tests/test_basis_table.py": (
        "test_table_values_and_gradients",
        "test_table_domain_and_point_limit",
        "test_table_round_trip",
        "test_float32_round_trip_keeps_validated_host_knots",
    ),
    "tests/test_bundled_basis_table.py": (
        "test_cold_cache_inside_jit_does_not_retain_tracers",
        "test_cache_keeps_float32_and_float64_coefficients_separate",
        "test_default_joint_separate_shapes_and_event_batch",
        "test_default_domain_edges_point_limit_and_no_fallback",
        "test_closed_ratio_endpoint_survives_coordinate_roundoff",
        "test_default_small_source_radius_gradients",
    ),
    "tests/test_joint_basis_table.py": (
        "test_small_source_fixed_separation_gradients",
        "test_unresolved_flux_retains_the_source_second_moment",
    ),
}
DEFAULT_IGNORED = (
    ".*",
    "*.egg",
    "_darcs",
    "build",
    "CVS",
    "dist",
    "node_modules",
    "venv",
    "{arch}",
)


def _patterns(value):
    return value.split() if isinstance(value, str) else list(value)


def discover_test_files():
    """Apply pytest's configured file patterns and test directories."""
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    options = configuration.get("tool", {}).get("pytest", {}).get("ini_options", {})
    patterns = _patterns(options.get("python_files", ["test_*.py", "*_test.py"]))
    ignored = _patterns(options.get("norecursedirs", DEFAULT_IGNORED))
    found = set()
    for relative in _patterns(options.get("testpaths", ["tests"])):
        test_root = (ROOT / relative).resolve()
        if not test_root.is_relative_to(ROOT) or not test_root.is_dir():
            raise ValueError(f"Expected a test directory within the checkout: {relative}")
        for directory, directories, filenames in os.walk(test_root, followlinks=False):
            directories[:] = sorted(
                name
                for name in directories
                if name != "__pycache__"
                and not any(fnmatch.fnmatchcase(name, pattern) for pattern in ignored)
                and not (Path(directory) / name).is_symlink()
            )
            for filename in filenames:
                if any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns):
                    path = Path(directory) / filename
                    if path.is_symlink():
                        raise ValueError(f"Test files must be stored in the checkout: {path}")
                    found.add(path.relative_to(ROOT).as_posix())
    if not found:
        raise ValueError("No test files match the configured pytest collection patterns.")
    return found


def partition_test_files(available):
    fixed = list(NUMERICAL_SHARDS.values())
    if fixed[0] & fixed[1]:
        raise ValueError("Numerical shards overlap.")
    shards = {name: paths & available for name, paths in NUMERICAL_SHARDS.items()}
    shards["general"] = available - set().union(*shards.values())
    members = [path for paths in shards.values() for path in paths]
    if len(members) != len(set(members)) or set(members) != available:
        raise ValueError("The partition must cover every test file exactly once.")
    return {name: sorted(paths) for name, paths in shards.items()}


def compatibility_selection(available):
    if COMPATIBILITY_FILES & COMPATIBILITY_CASES.keys():
        raise ValueError("Compatibility files and individual cases overlap.")
    selection = sorted(COMPATIBILITY_FILES & available)
    for path, cases in sorted(COMPATIBILITY_CASES.items()):
        if path in available:
            selection.extend(f"{path}::{case}" for case in cases)
    if len(selection) != len(set(selection)):
        raise ValueError("Compatibility selections overlap.")
    return selection


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", choices=[*NUMERICAL_SHARDS, "general", "compatibility"])
    parser.add_argument("--list", action="store_true", help="Print counts and paths as JSON")
    parser.add_argument("--junitxml", type=Path, help="Optional report below ignored local/")
    args = parser.parse_args(arguments)
    if args.suite is None and not args.list:
        parser.error("Choose a suite or use --list.")
    try:
        available = discover_test_files()
        shards = partition_test_files(available)
        compatibility = compatibility_selection(available)
        if args.list:
            print(
                json.dumps(
                    {
                        "total_test_files": len(available),
                        "complete_disjoint_partition": True,
                        "shards": {
                            name: {"file_count": len(paths), "paths": paths}
                            for name, paths in shards.items()
                        },
                        "compatibility": {
                            "selection_count": len(compatibility),
                            "paths": compatibility,
                        },
                    },
                    indent=2,
                )
            )
            return 0
        selected = compatibility if args.suite == "compatibility" else shards[args.suite]
        if not selected:
            parser.error(f"No available tests belong to {args.suite}.")
        command = [sys.executable, "-m", "pytest", "-v", "--durations=20", "--durations-min=1"]
        if args.junitxml is not None:
            report = (ROOT / args.junitxml).resolve()
            if not report.is_relative_to(ROOT / "local"):
                parser.error("Write JUnit reports below the ignored local/ directory.")
            report.parent.mkdir(parents=True, exist_ok=True)
            command.append(f"--junitxml={report}")
        print(f"Running {args.suite}: {len(selected)} file/case selections", flush=True)
        return subprocess.call([*command, *selected], cwd=ROOT)
    except (OSError, ValueError) as error:
        parser.exit(1, f"CI test partition failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
