"""Small plotting and provenance helpers shared by the examples."""

import hashlib
import importlib.metadata
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
from singlens import FFTConfig

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


def save_figure(fig, name, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", bbox_inches="tight")


def save_json(value, name, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"{name}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
