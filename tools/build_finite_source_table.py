"""Build an offline uniform/linear photometry and astrometry table.

Run from a source checkout with JAX, NumPy, and SciPy installed::

    PYTHONPATH=src python tools/build_finite_source_table.py --output /tmp/bases.npz

Optional --rho-grid and --z-grid files contain one numeric coordinate per line.
The output path is required; package data is never replaced implicitly. A
regenerated table needs accuracy checks before its use in precision inference.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import struct
from pathlib import Path

import numpy as np

GENERATOR_SETTINGS = {"n_disk": 96, "n_limb": 64, "split": True}
GENERATOR_FILES = (
    "lensing/concentric.py",
    "lensing/basis_table.py",
    "lensing/point.py",
)


def source_hashes():
    """Hash generator/interpolator source, using package-relative paths."""
    root = Path(importlib.util.find_spec("singlens").origin).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in GENERATOR_FILES
    }


def environment():
    """Record numerical dependencies without timestamps or machine identifiers."""
    return {
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "numpy", "scipy")
        },
    }


def numeric_payload_sha256(arrays):
    """Canonical hash of numeric NPZ members, independent of ZIP compression."""
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = arrays[name]
        if array.dtype.kind not in "biufc":
            continue
        dtype = array.dtype.newbyteorder("<")
        descriptor = json.dumps([name, dtype.str, list(array.shape)], separators=(",", ":")).encode(
            "utf-8"
        )
        digest.update(struct.pack("<Q", len(descriptor)))
        digest.update(descriptor)
        digest.update(np.ascontiguousarray(array, dtype=dtype).tobytes(order="C"))
    return digest.hexdigest()


def table_manifest(path, *, generation=None, validation=None):
    """Describe an existing NPZ; this does not evaluate or regenerate its bases."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    rho, z, coefficients = (arrays[name] for name in ("rho_grid", "z_grid", "coefficients"))
    if coefficients.dtype != np.dtype("float64") or coefficients.shape[-1] != 4:
        raise ValueError("Expected four float64 photometry/astrometry channels.")
    embedded = json.loads(str(arrays["metadata"]))
    return {
        "manifest_version": 1,
        "artifact": {
            "file": path.name,
            "bytes": path.stat().st_size,
            "zip_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "numeric_payload_sha256": numeric_payload_sha256(arrays),
            "numeric_payload_hash_format": (
                "Numeric members only, sorted by name. Each member hashes its UTF-8 JSON "
                "[name,little-endian dtype,shape] descriptor preceded by its uint64 little-endian "
                "byte length, then contiguous C-order little-endian array bytes."
            ),
            "format_version": int(arrays["format_version"]),
            "coordinate": str(arrays["coordinate"]),
            "representation": str(arrays["representation"]),
            "embedded_metadata": embedded,
        },
        "domain": {
            "rho": [float(rho[0]), float(rho[-1])],
            "u_over_rho": [float(z[0]), float(z[-1])],
            "a1": [0.0, 1.0],
            "endpoints_inclusive": True,
            "outside_domain": "NaN; rho=0 uses the exact point-source limit",
            "profiles": ["uniform", "linear"],
        },
        "grid": {
            "rho_count": len(rho),
            "u_over_rho_count": len(z),
            "coefficient_shape": list(coefficients.shape),
            "coefficient_bytes": coefficients.nbytes,
            "dtype": str(coefficients.dtype),
            "interpolation_coordinates": ["log(rho)", "log1p((u/rho)**2)"],
            "default_recipe": "geomspace(1e-6,10,192); joint_grid(refinement=2)",
        },
        "units": {
            "u_and_rho": "angular Einstein radius",
            "magnification": "dimensionless",
            "astrometric_shift": "angular Einstein radius; relative to unlensed source center",
        },
        "layout": {
            "npz_members": {
                name: {"shape": list(array.shape), "dtype": str(array.dtype)}
                for name, array in sorted(arrays.items())
            },
            "channels": ["L_uniform", "L_mu", "D_uniform", "D_mu"],
            "encoding": (
                "Spline coefficients for L_j=q*log(F_j/F_star), "
                "D_j=q*(C_j-C_star), q=1+(u/rho)**2, "
                "F_star=A_point(hypot(u,rho)), C_star=1/(u**2+2), C_j=shift_j/u. "
                "Mix unnormalized moments with flux weights when combining brightness bases."
            ),
        },
        "generation": generation
        or {
            "generator": "singlens.lensing.concentric.ConcentricDisk",
            "settings": GENERATOR_SETTINGS,
            "builder": "BasisTable.build(astrometry=True, batch_size=64)",
            "source_sha256": source_hashes(),
            "environment": environment(),
        },
        "validation": validation
        or {
            "status": "Not independently validated after this regeneration",
            "guidance": (
                "Use tools.validation.astrometry_methods for independent values and gradients. "
                "Custom grids do not inherit the bundled table's measured accuracy."
            ),
        },
        "reproduction": {
            "tool": "tools/build_finite_source_table.py",
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "command": (
                "PYTHONPATH=src python tools/build_finite_source_table.py --output /tmp/bases.npz"
            ),
            "notes": (
                "Optional --rho-grid/--z-grid files specify custom numeric grids. "
                "JSON has no generated timestamp. Bitwise numerical reproduction also depends "
                "on the recorded numerical-library versions and execution backend."
            ),
        },
    }


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination .npz file")
    parser.add_argument("--rho-grid", type=Path, help="Source radii, one numeric value per line")
    parser.add_argument("--z-grid", type=Path, help="u/rho ratios, one numeric value per line")
    parser.add_argument("--force", action="store_true", help="Replace the explicitly named outputs")
    args = parser.parse_args()
    manifest_path = args.output.with_suffix(".json")
    if args.output.suffix != ".npz":
        parser.error("--output must end in .npz")
    if not args.force and (args.output.exists() or manifest_path.exists()):
        parser.error("Output already exists; choose another path or use --force")

    import jax

    jax.config.update("jax_enable_x64", True)
    from singlens.lensing.basis_table import BasisTable
    from singlens.lensing.concentric import ConcentricDisk

    rho_grid = None if args.rho_grid is None else np.loadtxt(args.rho_grid)
    z_grid = None if args.z_grid is None else np.loadtxt(args.z_grid)
    table = BasisTable.build(
        ConcentricDisk(**GENERATOR_SETTINGS),
        rho_grid=rho_grid,
        z_grid=z_grid,
        astrometry=True,
        batch_size=64,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.save(args.output)
    manifest = table_manifest(args.output)
    manifest["generation"]["environment"]["backend"] = jax.default_backend()
    manifest["generation"]["environment"]["device_kind"] = jax.devices()[0].device_kind
    if args.rho_grid is not None or args.z_grid is not None:
        manifest["grid"]["default_recipe"] = None
    manifest["generation"]["custom_grid_sha256"] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (("rho", args.rho_grid), ("u_over_rho", args.z_grid))
        if path is not None
    }
    write_manifest(manifest_path, manifest)
    print(f"Wrote {args.output} and {manifest_path}")


if __name__ == "__main__":
    main()
