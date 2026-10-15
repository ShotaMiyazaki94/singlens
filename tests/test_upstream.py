"""Frozen parallax values from the original source checkout."""

import json
from pathlib import Path

import numpy as np

from singlens.trajectory import compute_parallax, set_parallax


def test_preserved_kepler_results():
    data = json.loads(Path(__file__).with_name("upstream_reference.json").read_text())
    p = data["parallax"]
    setup = set_parallax(8000.0, 0.0, 0.0, 266.0, -29.0)
    np.testing.assert_allclose(
        compute_parallax(np.array(p["t"]), 0.2, -0.1, setup), [p["d_tau"], p["d_beta"]], atol=1e-14
    )
