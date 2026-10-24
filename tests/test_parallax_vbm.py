"""Both benchmark engines must evaluate the same joint physical model."""

import os
import shlex
import shutil
import sys

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("emcee")
pytest.importorskip("numpyro")
pytest.importorskip("matplotlib")

from example.parallax_inference.code import experiment as exp
from example.parallax_inference.code import posterior as target
from example.parallax_inference.code.compare import physical, transformed
from example.parallax_inference.code.vbm import VBMPosterior


@pytest.fixture
def problem():
    pytest.importorskip("VBMicrolensing")
    if sys.platform not in {"darwin", "linux"}:
        pytest.skip("The array benchmark supports macOS and Linux")
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or not shutil.which(compiler[0]):
        pytest.skip("The array benchmark requires a C++17 compiler")
    times = np.array([-40.0, -0.35, -0.12, 0.0, 0.25, 60.0])
    durations = exp.exposure_schedule(times)
    geometry = exp.prepare_geometry(times, durations)
    truth = exp.CONFIG["event"]
    params = {name: truth[name] for name in target.NAMES}
    mean = (
        truth["f_s"]
        * exp.magnification_parameters(exp.make_model(), None, params, geometry=geometry)
        + truth["f_b"]
    )
    data = exp.Injection(
        jnp.asarray(times),
        mean + 0.001,
        0.01 * mean,
        params,
        mean,
        jnp.asarray(durations),
        geometry,
    )
    return data, VBMPosterior(data), target.pack(params)


def test_vbm_uses_same_parallax_exposures_and_sampled_limb_darkening(problem):
    data, vbm, theta = problem
    states = np.stack([theta, theta + np.array([0.01, 0.1, -0.05, 0.02, 0.03, -0.01, 0.8, 0, 0])])
    _, coefficients, positions = vbm.positions(states)
    for index, state in enumerate(states):
        expected = np.abs(exp.position(data.geometry, target.unpack(state)))
        np.testing.assert_allclose(positions[index], expected, rtol=1e-13, atol=1e-15)
        np.testing.assert_allclose(coefficients[index], target.unpack(state)["a1"])
    np.testing.assert_allclose(
        vbm.magnifications(states),
        vbm.magnifications(states, stock=True),
        rtol=1e-13,
    )
    assert coefficients[0] != coefficients[1]


def test_vbm_joint_density_matches_explicit_gaussian_and_keeps_both_fluxes(problem):
    data, vbm, theta = problem
    states = np.repeat(theta[None], 3, axis=0)
    states[1, 7] += 0.01
    states[2, 8] += 0.05
    values = vbm.magnifications(states)
    np.testing.assert_array_equal(values[0], values[1])
    actual = vbm(states)
    expected = []
    for state, magnification in zip(states, values):
        params = target.unpack(state)
        residual = (
            np.asarray(data.flux) - params["f_s"] * magnification - params["f_b"]
        ) / data.error
        log_likelihood = -0.5 * np.sum(
            residual**2 + np.log(2 * np.pi * np.asarray(data.error) ** 2)
        )
        log_prior = -0.5 * np.sum(((state - target.PRIOR_LOC) / target.PRIOR_SCALE) ** 2)
        expected.append(log_likelihood + log_prior)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    assert len(np.unique(actual)) == 3
    np.testing.assert_allclose(transformed(physical(states)), states, atol=1e-13)
    recovered = physical(states)
    for state, row in zip(states, recovered):
        np.testing.assert_allclose([target.unpack(state)[name] for name in target.NAMES], row)


def test_vbm_rejects_nonfinite_and_out_of_common_domain_proposals(problem):
    _, vbm, theta = problem
    states = np.repeat(theta[None], 4, axis=0)
    states[0, 3] = np.log(11.0)
    states[1, 7] = np.nan
    states[2, 2] = -1000.0
    states[3, 0] = 1e100
    assert np.all(np.isneginf(vbm(states)))


def test_spawned_vbm_workers_preserve_order_and_the_joint_density(problem):
    _, vbm, theta = problem
    states = np.repeat(theta[None], 7, axis=0)
    states[:, 6] += np.linspace(-1, 1, len(states))
    states[:, 7] += np.linspace(-0.02, 0.03, len(states))
    expected = vbm(states)
    try:
        vbm.start_workers(2)
        actual = vbm(states)
        np.testing.assert_array_equal(actual, expected)
        assert vbm.active_worker_cpu_seconds > 0
        assert vbm.metadata["parallel_execution"]["workers"] == 2
    finally:
        vbm.stop_workers()
    assert vbm.pool is None
