"""Persist enough NUTS state to diagnose a failed benchmark transition."""

import json
from importlib import import_module
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("numpyro")


def test_sampler_statistics_strict_json_preserves_nonfinite_locations():
    sampling = import_module("example.compare-single-vbml.code.sampling")
    fields = {
        "diverging": np.array([[False, True], [False, False]]),
        "num_steps": np.array([[3, 7], [3, 3]]),
        "potential_energy": np.array([[1.0, np.inf], [np.nan, 2.0]]),
    }
    report = sampling._sampler_statistics(fields)
    assert report["axes"] == ["chain", "draw"]
    assert report["warmup_included"] is False
    assert report["nonfinite_counts"]["potential_energy"] == 2
    assert report["values"]["potential_energy"] == [[1.0, None], [None, 2.0]]
    assert json.loads(json.dumps(report, allow_nan=False)) == report


@pytest.mark.parametrize("chains,method", [(2, "sequential"), (4, "vectorized"), (4, "parallel")])
def test_saved_nuts_statistics_match_draws_across_continuation_blocks(chains, method):
    import jax

    if method == "parallel" and jax.local_device_count() < chains:
        pytest.skip("Set JAX_NUM_CPU_DEVICES=4 to test parallel CPU chains")
    sampling = import_module("example.compare-single-vbml.code.sampling")
    initial = SimpleNamespace(x=np.array([0.4, -0.2]), hess_inv=np.array([[2.0, 0.3], [0.3, 1.0]]))
    draws, report = sampling.sample_nuts_until(
        lambda theta: 0.5 * jnp.dot(theta, theta),
        initial,
        lambda theta: {"x": theta[0], "y": theta[1]},
        warmup=50,
        samples=200,
        max_samples=200,
        seed=9,
        chains=chains,
        chain_method=method,
        check=lambda draws, diagnostics, fields: {"done": draws["x"].shape[1] >= 200},
    )
    fields = {
        name: np.asarray(value) for name, value in report["sampler_statistics"]["values"].items()
    }
    assert all(value.shape == (chains, 200) for value in fields.values())
    assert [block["draws_per_chain"] for block in report["sampling_history"]] == [100, 200]
    np.testing.assert_allclose(
        fields["potential_energy"], 0.5 * (draws["x"] ** 2 + draws["y"] ** 2), atol=1e-12
    )
    assert report["production_leapfrog_steps"] == fields["num_steps"].sum()
    assert report["divergences"] == fields["diverging"].sum()
    assert report["mean_accept_prob"] == fields["accept_prob"].mean()
    assert len(report["final_step_size"]) == chains
    assert np.all(np.asarray(report["final_step_size"]) > 0)
    assert not any(report["sampler_statistics"]["nonfinite_counts"].values())
    assert report["cpu_seconds_including_compile"] > 0
    assert report["warmup_cpu_seconds_including_compile"] > 0
    json.dumps(report, allow_nan=False)
