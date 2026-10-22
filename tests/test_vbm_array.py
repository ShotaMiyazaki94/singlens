"""The optimized benchmark binding must preserve the installed VBM numerics."""

import importlib
import json
import os
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.reference


@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("VBMicrolensing")
    if sys.platform not in {"darwin", "linux"}:
        pytest.skip("The array benchmark supports macOS and Linux")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("The array benchmark requires a C++17 compiler")
    return importlib.import_module("example.compare-single-vbml.code.vbm_array")


@pytest.mark.parametrize("tolerance,minannuli", [(1e-3, 2), (1e-6, 4)])
def test_array_matches_installed_scalar_core_exactly(adapter, tolerance, minannuli):
    import VBMicrolensing

    engine = VBMicrolensing.VBMicrolensing()
    engine.Tol = engine.RelTol = tolerance
    engine.minannuli = minannuli
    rho = np.array([0.005, 0.01, 0.02, 0.3])[::-1]
    a1 = np.array([0.0, 0.2, 0.8, 1.0])[::-1]
    # Exercise central crossing, source contact and the point-source shortcut.
    u = np.tile(np.r_[np.geomspace(1e-7, 3.0, 80), rho], (len(rho), 1))[:, ::-1]
    assert not u.flags.c_contiguous and not rho.flags.c_contiguous
    expected = []
    for radius, coefficient, row in zip(rho, a1, u):
        engine.a1 = float(coefficient)
        expected.append([engine.ESPLMag2(float(x), float(radius)) for x in row])
    with adapter.VBMArray(tolerance=tolerance, minannuli=minannuli) as bridge:
        np.testing.assert_array_equal(bridge.magnifications(rho, a1, u), expected)
        assert bridge.metadata["version"] == VBMicrolensing.__version__
        assert set(bridge.metadata["sha256"]) == {"header", "binary", "table", "adapter", "bridge"}
        assert bridge.metadata["tolerance"] == bridge.metadata["relative_tolerance"] == tolerance
        assert bridge.metadata["minannuli"] == minannuli
        assert all(path.startswith("VBMicrolensing/") for path in bridge.metadata["files"].values())
        command = bridge.metadata["build_command"]
        assert not Path(command[0]).is_absolute()
        assert command[command.index("-I") + 1] == "<VBMicrolensing>/lib"
        assert command[-3] == "vbm_array.cpp"
        assert command[-1].startswith("<build-dir>/")
        serialized = json.dumps(bridge.metadata)
        for private_path in (
            Path(VBMicrolensing.__file__).resolve().parent,
            Path(adapter.__file__).resolve().parent,
            adapter._load_bridge()[2].name,
        ):
            assert str(private_path) not in serialized


def test_instances_reuse_build_but_not_settings(adapter, monkeypatch):
    with adapter.VBMArray() as first:

        def no_compilation(*args, **kwargs):
            raise AssertionError("A second VBMArray must reuse the loaded bridge")

        monkeypatch.setattr(adapter.subprocess, "run", no_compilation)
        with adapter.VBMArray(tolerance=1e-6, minannuli=4) as second:
            assert first._lib is second._lib
            assert first.metadata["tolerance"] == 1e-3
            assert second.metadata["tolerance"] == 1e-6
            second.metadata["sha256"]["header"] = "edited"
            assert first.metadata["sha256"]["header"] != "edited"
    first.close()
    with pytest.raises(RuntimeError, match="closed"):
        first.magnifications([0.01], [0.5], [[0.1]])


@pytest.mark.parametrize("shape", [(0, 3), (2, 0)])
def test_empty_arrays(adapter, shape):
    with adapter.VBMArray() as bridge:
        actual = bridge.magnifications(np.full(shape[0], 0.01), np.zeros(shape[0]), np.empty(shape))
    assert actual.shape == shape and actual.dtype == np.float64


@pytest.mark.parametrize(
    "rho,a1,u",
    [
        ([0.01], [0.5], [0.1, 0.2]),
        (0.01, [0.5], [[0.1, 0.2]]),
        ([0.01, 0.02], [0.5], [[0.1, 0.2]]),
    ],
)
def test_shape_errors(adapter, rho, a1, u):
    with adapter.VBMArray() as bridge, pytest.raises(ValueError, match="shape"):
        bridge.magnifications(rho, a1, u)


@pytest.mark.parametrize(
    "rho,a1,u",
    [
        (0.0, 0.5, 0.1),
        (-0.01, 0.5, 0.1),
        (np.nan, 0.5, 0.1),
        (np.inf, 0.5, 0.1),
        (101.0, 0.5, 0.1),
        (1e-200, 0.5, 0.1),
        (0.01, -0.1, 0.1),
        (0.01, 1.1, 0.1),
        (0.01, np.nan, 0.1),
        (0.01, 0.5, -0.1),
        (0.01, 0.5, np.nan),
        (0.01, 0.5, np.inf),
    ],
)
def test_invalid_proposals_produce_nan_row_only(adapter, rho, a1, u):
    with adapter.VBMArray() as bridge:
        values = bridge.magnifications([rho, 0.01], [a1, 0.5], [[0.1, u], [0.1, 0.2]])
    assert np.isnan(values[0]).all()
    assert np.isfinite(values[1]).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"tolerance": 0},
        {"tolerance": np.nan},
        {"tolerance": np.inf},
        {"minannuli": 0},
        {"minannuli": 2.5},
        {"minannuli": True},
        {"minannuli": 2**40},
    ],
)
def test_invalid_settings(adapter, settings):
    with pytest.raises(ValueError):
        adapter.VBMArray(**settings)


def test_mismatched_wheel_files_are_rejected_before_loading(adapter, monkeypatch):
    monkeypatch.setattr(adapter, "_sha256", lambda path: "00" * 32)
    with pytest.raises(RuntimeError, match="RECORD"):
        adapter._installed_files()
