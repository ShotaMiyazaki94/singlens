#!/usr/bin/env python3
"""Acquire the CK04 spectra locally; do not redistribute the extracted CSV.

The checked-in provenance records the original STScI URLs and SHA256 values.
Only acquisition needs astropy. ``--bootstrap`` can install its pinned version
in a temporary isolated environment without changing the runtime environment.
An existing, verified CSV needs neither astropy nor a network connection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "example" / "inference" / "data"
PROVENANCE = DATA / "active_reference_provenance.json"
CSV_NAME = "ck04_5000_4000_logg4p5_solar.csv"
FITS_NAMES = ("ckp00_5000.fits", "ckp00_4000.fits")
ACQUISITION_DEPENDENCY = "astropy==8.0.1"
HEADER = "wavelength_angstrom,photosphere_surface_flux,spot_surface_flux\n"


def _verify(payload, expected, name):
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {name}: expected {expected}, got {actual}")
    return payload


def _existing(target, expected):
    if not target.exists():
        return False
    try:
        _verify(target.read_bytes(), expected, target.name)
    except ValueError as error:
        raise ValueError(f"{error}. Move the existing file aside before retrying.") from error
    return True


def _merge_spectra(photosphere, spot):
    """Join matching native samples in Angstrom without interpolation."""
    selected = []
    for spectrum in (photosphere, spot):
        samples = [(float(w), float(f)) for w, f in spectrum if 7500 <= float(w) <= 23000]
        if not samples or not all(math.isfinite(w) and math.isfinite(f) for w, f in samples):
            raise ValueError("CK04 spectra must contain finite native samples over 7500–23000 A.")
        if any(left[0] >= right[0] for left, right in zip(samples, samples[1:])):
            raise ValueError("CK04 native wavelength samples must be strictly increasing.")
        selected.append(samples)
    if [w for w, _ in selected[0]] != [w for w, _ in selected[1]]:
        raise ValueError("CK04 native wavelength grids differ; no resampling is permitted.")
    rows = [HEADER]
    for (wavelength, photo_flux), (_, spot_flux) in zip(*selected):
        rows.append(
            ",".join(format(value, ".14g") for value in (wavelength, photo_flux, spot_flux)) + "\n"
        )
    return "".join(rows).encode("ascii")


def _extract_spectra(raw):
    # Deliberately lazy: ordinary tests and singlens itself do not need astropy.
    try:
        from astropy.io import fits
    except ImportError as error:
        raise RuntimeError(
            "Reading the original CK04 FITS files needs acquisition-only astropy. "
            "Run python tools/fetch_example_data.py --bootstrap."
        ) from error
    spectra = []
    for name in FITS_NAMES:
        with fits.open(io.BytesIO(raw[name]), memmap=False) as hdus:
            table = hdus[1].data
            spectra.append(list(zip(table["WAVELENGTH"], table["g45"])))
    return _merge_spectra(*spectra)


def _fetch_ck04(entry, output_dir, raw_dir=None):
    target = output_dir / CSV_NAME
    if _existing(target, entry["extracted_sha256"]):
        return target
    urls = {Path(urllib.parse.urlsplit(url).path).name: url for url in entry["urls"]}
    raw = {}
    for name in FITS_NAMES:
        if raw_dir is not None:
            payload = (raw_dir / name).read_bytes()
        else:
            url = urls[name]
            if urllib.parse.urlsplit(url).scheme != "https":
                raise ValueError("CK04 downloads require the HTTPS URLs in the provenance file.")
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = response.read()
        raw[name] = _verify(payload, entry["raw_sha256"][name], name)
    payload = _verify(_extract_spectra(raw), entry["extracted_sha256"], CSV_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Never replace a user's local input, including a concurrent acquisition.
    with target.open("xb") as stream:
        stream.write(payload)
    return target


def _bootstrap(arguments):
    with tempfile.TemporaryDirectory(prefix="singlens-example-data-") as temporary:
        environment = Path(temporary) / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                ACQUISITION_DEPENDENCY,
            ],
            check=True,
        )
        subprocess.run(
            [
                str(python),
                str(Path(__file__).resolve()),
                *(argument for argument in arguments if argument != "--bootstrap"),
            ],
            check=True,
        )


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA,
        help="Local CSV destination (default: example/inference/data).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Use local ckp00_5000.fits and ckp00_4000.fits without downloading atmosphere data.",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help=f"If astropy is absent, install {ACQUISITION_DEPENDENCY} in a temporary isolated venv.",
    )
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    args = parser.parse_args(arguments)
    try:
        entry = json.loads(PROVENANCE.read_text(encoding="utf-8"))["inputs"][CSV_NAME]
        target = args.output_dir / CSV_NAME
        if _existing(target, entry["extracted_sha256"]):
            print(f"Verified existing CK04 input: {target}")
            return 0
        if importlib.util.find_spec("astropy") is None:
            if not args.bootstrap:
                parser.error(
                    "Acquisition-only astropy is absent. Retry with --bootstrap to use a temporary "
                    "isolated environment; singlens runtime dependencies stay unchanged."
                )
            _bootstrap(arguments)
            return 0
        target = _fetch_ck04(entry, args.output_dir, args.raw_dir)
        print(f"Acquired and verified CK04 input: {target}")
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"CK04 acquisition failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
