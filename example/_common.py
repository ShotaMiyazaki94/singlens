"""Small plotting and provenance helpers shared by the examples."""

import hashlib
import importlib.metadata
import importlib.resources
import importlib.util
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import jax
import matplotlib.pyplot as plt

import singlens
from singlens.lensing import BasisTable, FFTConfig

COLORS = ["#247BA0", "#E07A5F", "#3D8067", "#8765A5", "#C59A32"]


@cache
def hardware_metadata():
    """Record only the machine name and chip, without device identifiers."""
    hardware = {}
    if processor := platform.processor():
        hardware["chip"] = processor
    if platform.system() != "Darwin":
        return hardware

    try:
        result = subprocess.run(
            ["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        records = json.loads(result.stdout).get("SPHardwareDataType", [])
        details = records[0] if records else {}
        for source, target in (
            ("machine_name", "machine_name"),
            ("chip_type" if details.get("chip_type") else "cpu_type", "chip"),
        ):
            if isinstance(value := details.get(source), str) and value:
                hardware[target] = value
    except (
        OSError,
        subprocess.SubprocessError,
        ValueError,
        AttributeError,
        TypeError,
        KeyError,
        RuntimeError,
    ):
        pass
    return hardware


def style():
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.top": False,
            "ytick.right": False,
            "axes.prop_cycle": plt.cycler(color=COLORS),
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def metadata():
    packages = ["singlens", "jax", "jaxlib", "numpy", "scipy", "matplotlib", "numpyro"]
    return {
        "versions": {
            name: importlib.metadata.version(name)
            for name in packages
            if importlib.util.find_spec(name) is not None
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {
            str(p.relative_to(Path(singlens.__file__).parent)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(Path(singlens.__file__).parent.rglob("*.py"))
        },
        "python": platform.python_version(),
        "platform": platform.platform(),
        "hardware": dict(hardware_metadata()),
        "devices": [str(device) for device in jax.devices()],
        "device_kind": jax.devices()[0].device_kind,
        "x64": jax.config.jax_enable_x64,
        "fft_defaults": asdict(FFTConfig()),
    }


@cache
def _bundled_table_provenance():
    resources = importlib.resources.files("singlens.data")
    manifest_bytes = resources.joinpath("finite_source_table.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    artifact = manifest["artifact"]
    checksum = hashlib.sha256(resources.joinpath(artifact["file"]).read_bytes()).hexdigest()
    if checksum != artifact["zip_sha256"]:
        raise RuntimeError("Bundled finite-source table does not match its provenance manifest")
    return {
        "artifact": {
            key: artifact[key]
            for key in ("file", "zip_sha256", "numeric_payload_sha256", "format_version")
        }
        | {"manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()},
        "generation": manifest["generation"],
        "validation_report": manifest["validation"]["report"],
    }


def model_solver_metadata(model):
    """Record the solver selected on a Model, including packaged-table provenance."""
    solver = model.solver
    result = {"source": model.source, "configuration": {}}
    if model.source == "point":
        return result | {"solver": "point"}
    if isinstance(solver, FFTConfig):
        return result | {"solver": "fft", "configuration": asdict(solver)}
    if not isinstance(solver, BasisTable):
        raise TypeError("Expected a Model with an FFTConfig or BasisTable solver")
    from singlens.lensing.basis_table import _bundled_basis_table

    bundled = solver is _bundled_basis_table(jax.config.x64_enabled)
    result.update(
        solver="table",
        origin="bundled" if bundled else "custom",
        configuration={
            "rho": [float(solver.rho_grid[0]), float(solver.rho_grid[-1])],
            "u_over_rho": [float(solver.z_grid[0]), float(solver.z_grid[-1])],
            "rho_count": len(solver.rho_grid),
            "u_over_rho_count": len(solver.z_grid),
            "coefficient_shape": list(solver._coefficients.shape),
            "dtype": str(solver._coefficients.dtype),
            "coordinate": solver.coordinate,
            "representation": solver.representation,
            "has_astrometry": solver.has_astrometry,
            "outside_domain": "NaN; rho=0 uses the analytic point-source limit",
        },
    )
    if bundled:
        result.update(_bundled_table_provenance())
    return result


def save_figure(fig, name, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")


def save_json(value, name, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
