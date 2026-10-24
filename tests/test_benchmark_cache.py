"""Reject stale or invalid independent-reference artifacts before any inference."""

import copy
import json
from importlib import import_module

import numpy as np
import pytest

pytestmark = pytest.mark.reference
pytest.importorskip("matplotlib")


@pytest.fixture
def reference_cache(tmp_path, monkeypatch):
    experiment = import_module("example.compare-single-vbml.code.experiment")
    times = np.array([-1.0, 0.0, 1.0])
    mean = np.array([1.01, 2.0, 1.01])
    error = experiment.photometric_error(mean, experiment.CONFIG)
    monkeypatch.setattr(experiment, "observation_times", lambda: times)
    checks = {
        "saturation": experiment.saturation_check(mean, experiment.CONFIG),
        "selected_epochs": [0, 1, 2],
        "injection_refinement_max_flux_error_sigma": 0.0,
        "fitting_exposure_rule_max_flux_error_sigma": 0.0,
        "overlap_max_relative_difference": 0.0,
        "injection_order": experiment.INJECTION_ORDER,
        "refined_exposure_order": 15,
        "wing_orders": [32, 64],
        "wing_boundary_u_over_rho": 3,
    }
    path = tmp_path / "reference.npz"
    experiment.save_reference_data(path, times=times, error=error, lens_flux=mean, checks=checks)
    return experiment, path, times, error, mean, checks


def rewrite_reference(path, change):
    """Tamper one field without accidentally changing another serialized field."""
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in ("times", "error", "lens_flux")}
        description = json.loads(str(saved["metadata"]))
    change(arrays, description)
    np.savez_compressed(path, **arrays, metadata=json.dumps(description, sort_keys=True))


def test_reference_cache_round_trip_preserves_numeric_and_validation_provenance(reference_cache):
    experiment, path, times, error, mean, checks = reference_cache
    loaded_times, loaded_error, loaded_mean, loaded_checks = experiment.load_reference_data(path)
    for actual, expected in zip((loaded_times, loaded_error, loaded_mean), (times, error, mean)):
        np.testing.assert_array_equal(actual, expected)
    assert loaded_checks == checks
    with np.load(path, allow_pickle=False) as saved:
        description = json.loads(str(saved["metadata"]))
    assert description["numeric_payload_sha256"] == experiment.reference_data_hash(
        times, error, mean
    )
    provenance = description["provenance"]
    assert provenance["injection_exposure_order"] == experiment.INJECTION_ORDER
    assert provenance["fit_exposure_order"] == experiment.EXPOSURE_ORDER
    assert set(provenance["source_sha256"]) >= {
        "integral",
        "_wing_integral",
        "reference_magnification",
        "exposure_rule",
        "photometric_error",
    }
    assert all(len(digest) == 64 for digest in provenance["source_sha256"].values())


def test_reference_injection_is_independent_of_fitted_flux_prior(reference_cache, monkeypatch):
    experiment, path, times, error, mean, _ = reference_cache
    before = experiment.synthetic_data(seed=43, reference=path)
    monkeypatch.setattr(experiment, "PRIOR_LOC", experiment.PRIOR_LOC + 0.1)
    monkeypatch.setattr(experiment, "FLUX_PRIOR_MEAN", np.array([0.5, 0.5]))

    def unexpected_likelihood(*args, **kwargs):
        raise AssertionError("Synthetic data must not depend on a fitting likelihood")

    monkeypatch.setattr(experiment, "flux_conditional", unexpected_likelihood)
    after = experiment.synthetic_data(seed=43, reference=path)
    assert experiment.data_hash(before) == experiment.data_hash(after)
    for actual, expected in zip((after.times, after.error, after.lens_flux), (times, error, mean)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("field", ["source", "configuration", "exposure_order"])
def test_reference_cache_rejects_different_generation_provenance(reference_cache, field):
    experiment, path, *_ = reference_cache

    def alter(_arrays, description):
        provenance = description["provenance"]
        if field == "source":
            provenance["source_sha256"]["integral"] = "0" * 64
        elif field == "configuration":
            provenance["configuration_sha256"] = "0" * 64
        else:
            provenance["injection_exposure_order"] += 2

    rewrite_reference(path, alter)
    with pytest.raises(ValueError, match="configuration|source|provenance"):
        experiment.load_reference_data(path)


def test_reference_cache_rejects_changed_current_physics(reference_cache, monkeypatch):
    experiment, path, *_ = reference_cache
    changed = copy.deepcopy(experiment.CONFIG)
    changed["source_radius_solar"] *= 1.01
    monkeypatch.setattr(experiment, "CONFIG", changed)
    with pytest.raises(ValueError, match="configuration|source|provenance"):
        experiment.synthetic_data(reference=path)


def test_reference_cache_rejects_changed_current_reference_source(reference_cache, monkeypatch):
    experiment, path, *_ = reference_cache
    accuracy = import_module("example.compare-single-vbml.code.accuracy")

    def changed_integral(*args, **kwargs):
        raise AssertionError("A stale reference must be rejected before recomputation")

    monkeypatch.setattr(accuracy, "integral", changed_integral)
    with pytest.raises(ValueError, match="configuration|source|provenance"):
        experiment.synthetic_data(reference=path)


def test_reference_cache_rejects_checksum_corruption(reference_cache):
    experiment, path, *_ = reference_cache

    def alter(arrays, _description):
        arrays["lens_flux"][1] += 0.01

    rewrite_reference(path, alter)
    with pytest.raises(ValueError, match="checksum"):
        experiment.load_reference_data(path)


def test_reference_cache_rejects_different_times_even_with_valid_checksum(reference_cache):
    experiment, path, *_ = reference_cache

    def alter(arrays, description):
        arrays["times"] += 0.001
        description["numeric_payload_sha256"] = experiment.reference_data_hash(
            arrays["times"], arrays["error"], arrays["lens_flux"]
        )

    rewrite_reference(path, alter)
    with pytest.raises(ValueError, match="observation times"):
        experiment.load_reference_data(path)


def test_reference_cache_rejects_broadcastable_error_shape(reference_cache):
    experiment, path, *_ = reference_cache

    def alter(arrays, description):
        # This would broadcast an N-epoch likelihood to N x N; its flattened
        # numeric checksum alone does not detect the shape change.
        arrays["error"] = arrays["error"][:, None]
        description["numeric_payload_sha256"] = experiment.reference_data_hash(
            arrays["times"], arrays["error"], arrays["lens_flux"]
        )

    rewrite_reference(path, alter)
    with pytest.raises(ValueError, match="shape|dimension|invalid"):
        experiment.load_reference_data(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("times", np.nan),
        ("error", np.inf),
        ("lens_flux", np.nan),
        ("error", 0.0),
        ("lens_flux", -1.0),
    ],
)
def test_reference_cache_rejects_invalid_payload_even_with_valid_checksum(
    reference_cache, field, value
):
    experiment, path, *_ = reference_cache

    def alter(arrays, description):
        arrays[field][1] = value
        description["numeric_payload_sha256"] = experiment.reference_data_hash(
            arrays["times"], arrays["error"], arrays["lens_flux"]
        )

    rewrite_reference(path, alter)
    with pytest.raises(ValueError, match="finite|positive|invalid"):
        experiment.load_reference_data(path)


@pytest.mark.parametrize(
    "field,value",
    [
        ("injection_refinement_max_flux_error_sigma", 1.0),
        ("fitting_exposure_rule_max_flux_error_sigma", np.nan),
        ("overlap_max_relative_difference", 1.0),
        ("saturation", {"passed": False}),
    ],
)
def test_reference_cache_rejects_failed_validation(reference_cache, field, value):
    experiment, path, *_ = reference_cache

    def alter(_arrays, description):
        description["checks"][field] = value

    rewrite_reference(path, alter)
    with pytest.raises(ValueError, match="validation|check|saturation|reference"):
        experiment.load_reference_data(path)
