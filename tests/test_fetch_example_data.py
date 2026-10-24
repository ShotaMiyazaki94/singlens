"""Local acquisition uses verified originals without bundling atmosphere data."""

import hashlib
import json

import pytest

from tools import fetch_example_data as fetch


def test_native_spectra_extraction_preserves_samples_and_range():
    # Synthetic values, independent of the externally hosted CK04 spectra.
    photosphere = [(7490, 1), (7500, 2.25), (10000, 3), (23000, 4), (23100, 5)]
    spot = [(wavelength, flux / 2) for wavelength, flux in photosphere]
    assert fetch._merge_spectra(photosphere, spot) == (
        fetch.HEADER + "7500,2.25,1.125\n10000,3,1.5\n23000,4,2\n"
    ).encode("ascii")
    with pytest.raises(ValueError, match="no resampling"):
        fetch._merge_spectra(photosphere, [(7500, 1), (11000, 2), (23000, 3)])


def test_verified_existing_input_uses_no_network_or_acquisition_dependency(tmp_path, monkeypatch):
    payload = b"synthetic local input\n"
    (tmp_path / fetch.CSV_NAME).write_bytes(payload)
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {"inputs": {fetch.CSV_NAME: {"extracted_sha256": hashlib.sha256(payload).hexdigest()}}}
        )
    )
    monkeypatch.setattr(fetch, "PROVENANCE", provenance)

    def forbidden(*args, **kwargs):
        raise AssertionError("A verified local CSV must not acquire dependencies or data")

    monkeypatch.setattr(fetch.importlib.util, "find_spec", forbidden)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", forbidden)
    assert fetch.main(["--output-dir", str(tmp_path), "--bootstrap"]) == 0


def test_corrupt_original_is_rejected_before_extraction(tmp_path, monkeypatch):
    raw_dir, output_dir = tmp_path / "raw", tmp_path / "output"
    raw_dir.mkdir()
    expected = b"synthetic original"
    (raw_dir / fetch.FITS_NAMES[0]).write_bytes(b"corrupt original")
    entry = dict(
        urls=[f"https://example.invalid/{name}" for name in fetch.FITS_NAMES],
        raw_sha256={name: hashlib.sha256(expected).hexdigest() for name in fetch.FITS_NAMES},
        extracted_sha256="unused",
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Unverified originals must not be extracted")

    monkeypatch.setattr(fetch, "_extract_spectra", forbidden)
    with pytest.raises(ValueError, match="SHA256 mismatch for ckp00_5000.fits"):
        fetch._fetch_ck04(entry, output_dir, raw_dir)
    assert not output_dir.exists()


def test_existing_modified_input_is_preserved(tmp_path):
    target = tmp_path / fetch.CSV_NAME
    target.write_bytes(b"local modified input")
    with pytest.raises(ValueError, match="Move the existing file aside"):
        fetch._fetch_ck04({"extracted_sha256": "different"}, tmp_path)
    assert target.read_bytes() == b"local modified input"
