"""Benchmark-only NumPy binding to the *installed* VBM numerical binary.

Only the small array adapter is compiled; ESPLMag2, its constructor, and its
table loader remain the installed wheel's implementations. A matching wheel
header and exported C++ symbols are required. The bridge is built once per
process in a fresh temporary directory, so cold setup includes compilation.

PyDLL holds the Python GIL during calls. This is intentional: VBM's scalar
routine has shared static scratch state and must not run concurrently.
"""

from __future__ import annotations

import base64
import copy
import ctypes
import hashlib
import inspect
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from functools import cache
from importlib.metadata import distribution
from pathlib import Path
from time import perf_counter

import numpy as np

_BUFFER = np.ctypeslib.ndpointer(dtype=np.float64, flags=("C_CONTIGUOUS", "ALIGNED"))
_BUILD_LOCK = threading.Lock()


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _installed_files():
    import VBMicrolensing

    package = Path(VBMicrolensing.__file__).resolve().parent
    paths = {
        "binary": Path(inspect.getfile(VBMicrolensing.VBMicrolensing)).resolve(),
        "header": package / "lib" / "VBMicrolensingLibrary.h",
        "table": package / "data" / "ESPL.tbl",
    }
    dist = distribution("VBMicrolensing")
    if dist.version.split(".")[0] != "5":
        raise RuntimeError(
            f"The array benchmark supports VBMicrolensing 5.x, found {dist.version}. "
            "Install a compatible wheel or explicitly select the stock binding."
        )
    records = {Path(dist.locate_file(item)).resolve(): item for item in dist.files or []}
    hashes = {}
    for kind, path in paths.items():
        if not path.is_file() or not path.is_relative_to(package):
            raise RuntimeError(
                f"VBM array binding needs {kind} from the same installed wheel: {path}. "
                "Reinstall the VBMicrolensing wheel including lib/ and data/."
            )
        record = records.get(path)
        digest = _sha256(path)
        expected = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode()
        if (
            record is None
            or record.hash is None
            or record.hash.mode != "sha256"
            or record.hash.value != expected
        ):
            raise RuntimeError(
                f"Cannot verify VBM {kind} against this wheel's RECORD: {path}. "
                "Reinstall an unmodified VBMicrolensing wheel; a mismatched C++ "
                "header and binary can corrupt memory."
            )
        hashes[kind] = digest
    return dist.version, paths, hashes


@cache
def _load_bridge():
    if sys.platform not in {"darwin", "linux"}:
        raise RuntimeError(
            "VBM array binding requires macOS or Linux with a C++17 compiler. "
            "Use the explicitly selected stock binding on other platforms."
        )
    version, paths, hashes = _installed_files()
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        raise RuntimeError(
            "VBM array binding needs a C++17 compiler. Install Xcode command line "
            "tools on macOS or a C++ toolchain on Linux, or set CXX."
        )
    compiler[0] = shutil.which(compiler[0])
    source = Path(__file__).with_suffix(".cpp")
    if not source.is_file():
        raise RuntimeError(f"Missing VBM adapter source: {source}. Restore the example sources.")
    # Keep this object alive alongside the loaded library until process exit.
    directory = tempfile.TemporaryDirectory(prefix="singlens-vbm-array-")
    library = Path(directory.name) / ("bridge.dylib" if sys.platform == "darwin" else "bridge.so")
    command = compiler + ["-O3", "-DNDEBUG", "-std=c++17", "-fPIC"]
    command += (
        ["-dynamiclib", "-undefined", "dynamic_lookup"] if sys.platform == "darwin" else ["-shared"]
    )
    command += ["-I", str(paths["header"].parent), str(source), "-o", str(library)]
    try:
        compiler_version = subprocess.run(
            compiler + ["--version"], check=True, capture_output=True, text=True, timeout=30
        ).stdout.splitlines()[0]
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        # Promote the original wheel's C++ symbols for the unresolved bridge.
        original = ctypes.PyDLL(str(paths["binary"]), mode=ctypes.RTLD_GLOBAL)
        lib = ctypes.PyDLL(str(library))
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(
            "Could not build/load the VBM array adapter. It requires a C++17 "
            "compiler compatible with the installed wheel and exported VBM C++ "
            "symbols. Reinstall a compatible VBMicrolensing 5.x wheel or "
            f"explicitly select the stock binding. Details: {detail}"
        ) from exc
    lib.vbm_array_create.argtypes = [ctypes.c_char_p, ctypes.c_double, ctypes.c_int]
    lib.vbm_array_create.restype = ctypes.c_void_p
    lib.vbm_array_destroy.argtypes = [ctypes.c_void_p]
    lib.vbm_array_destroy.restype = None
    lib.vbm_array_magnifications.argtypes = [
        ctypes.c_void_p,
        _BUFFER,
        _BUFFER,
        _BUFFER,
        ctypes.c_size_t,
        ctypes.c_size_t,
        _BUFFER,
    ]
    lib.vbm_array_magnifications.restype = ctypes.c_int
    package = paths["header"].parents[1]
    published_command = [
        part.replace(str(package), "<VBMicrolensing>")
        .replace(str(source), source.name)
        .replace(directory.name, "<build-dir>")
        for part in command
    ]
    published_command[0] = Path(compiler[0]).name
    metadata = {
        "binding": "contiguous-array",
        "numerical_implementation": "installed-wheel ESPLMag2 (not recompiled)",
        "version": version,
        "platform": sys.platform,
        "machine": platform.machine(),
        "compiler": compiler_version,
        "build_command": published_command,
        "files": {
            key: str(Path("VBMicrolensing") / path.relative_to(package))
            for key, path in paths.items()
        },
        "sha256": {**hashes, "adapter": _sha256(source), "bridge": _sha256(library)},
        "abi_check": "5.x header, binary and table verified against same distribution RECORD",
        "serialization": "ctypes.PyDLL retains Python GIL",
        "build_cache": "once per process; fresh temporary build",
    }
    return lib, original, directory, paths["table"], metadata


class VBMArray:
    """Evaluate ``u[walkers, epochs]`` using per-walker ``rho`` and ``a1``.

    Noncontiguous input is copied to aligned float64 buffers. Empty batches are
    supported. Invalid physical values produce an all-NaN row, allowing the
    posterior to reject proposals. The numerical guard requires finite values,
    1e-150 <= rho <= 100 (VBM's table limit), 0 <= a1 <= 1, and 0 <= u <= 1e150.
    Shape errors raise ValueError. This helper is only for the benchmark's
    linear limb-darkening profile, with astrometry disabled.
    """

    def __init__(self, tolerance=1e-3, minannuli=2):
        started = perf_counter()
        self._handle = None
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be finite and positive")
        if (
            isinstance(minannuli, bool)
            or not isinstance(minannuli, (int, np.integer))
            or not 1 <= minannuli <= np.iinfo(np.int32).max
        ):
            raise ValueError("minannuli must be a positive C int")
        with _BUILD_LOCK:
            lib, _, _, table, metadata = _load_bridge()
        self._lib = lib
        self._handle = lib.vbm_array_create(os.fsencode(table), tolerance, minannuli)
        if not self._handle:
            raise RuntimeError("The installed VBM wheel could not construct its ESPL engine.")
        self.setup_seconds = perf_counter() - started
        self.metadata = copy.deepcopy(metadata)
        self.metadata.update(
            {
                "tolerance": float(tolerance),
                "relative_tolerance": float(tolerance),
                "minannuli": int(minannuli),
                "setup_seconds": self.setup_seconds,
            }
        )

    def magnifications(self, rho, a1, u):
        if not self._handle:
            raise RuntimeError("VBMArray is closed")
        rho, a1, u = [
            np.require(value, dtype=np.float64, requirements=["C", "A"]) for value in (rho, a1, u)
        ]
        if u.ndim != 2 or rho.shape != (u.shape[0],) or a1.shape != rho.shape:
            raise ValueError(
                "u must have shape (walkers, epochs); rho/a1 must have shape (walkers,)"
            )
        out = np.empty(u.shape, dtype=np.float64)
        if out.size:
            status = self._lib.vbm_array_magnifications(self._handle, rho, a1, u, *u.shape, out)
            if status:
                raise RuntimeError("The installed VBM ESPLMag2 routine raised a C++ exception.")
        return out

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.vbm_array_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        if not self._handle:
            raise RuntimeError("VBMArray is closed")
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()
