"""Inference commands must reject failed optimization and sampling diagnostics."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

pytest.importorskip("numpyro")
pytest.importorskip("tinygp")

from example.inference.code import active_ffp as active
from example.roman_astrometry.code import fit as roman


def optimizer_result(size, *, objective, success=True, finite_gradient=True):
    return OptimizeResult(
        x=np.zeros(size),
        fun=objective,
        jac=np.zeros(size) if finite_gradient else np.full(size, np.nan),
        success=success,
        nit=2,
        message="Synthetic optimizer outcome",
    )


def fit_problem(kind):
    if kind == "active":
        problem = SimpleNamespace(
            times=np.array([-1.0, 0.0, 1.0]),
            flux=np.array([1.0, 3.0, 1.0]),
            prior_mean=np.zeros(9),
            prior_sd=np.ones(9),
            objective=None,
        )
        return active, problem, active.ActiveFFPFit.fit_map, 9
    problem = SimpleNamespace(sign=1, objective=None, initial=lambda k: np.zeros(8))
    return roman, problem, roman.RomanJointFit.fit_map, 8


@pytest.mark.parametrize("kind", ["active", "roman"])
def test_map_selection_ignores_failed_and_nonfinite_optimizers(kind, monkeypatch):
    module, problem, fit, size = fit_problem(kind)
    failed = optimizer_result(size, objective=-100.0, success=False)
    nonfinite = optimizer_result(size, objective=-10.0, finite_gradient=False)
    valid = optimizer_result(size, objective=1.0)
    results = iter([failed, nonfinite, valid])
    monkeypatch.setattr(module, "minimize", lambda *args, **kwargs: next(results))
    best, records = fit(problem, starts=3)
    assert best is valid
    assert [record["success"] for record in records] == [False, True, True]
    assert [record["finite"] for record in records] == [True, False, True]


@pytest.mark.parametrize("kind", ["active", "roman"])
def test_map_failure_stops_before_hessian_or_sampling(kind, monkeypatch):
    module, problem, fit, size = fit_problem(kind)
    monkeypatch.setattr(
        module,
        "minimize",
        lambda *args, **kwargs: optimizer_result(size, objective=np.inf, success=False),
    )
    with pytest.raises(RuntimeError, match="No finite successful"):
        fit(problem, starts=2)


@pytest.mark.parametrize("failed_branch", ["positive", "negative"])
def test_roman_failed_convergence_persists_diagnostics_and_raises(
    tmp_path, monkeypatch, failed_branch
):
    class Problem:
        def __init__(self, *, seed, u0_sign):
            self.seed, self.sign = seed, u0_sign
            self.settings = dict(seed=seed, u0_sign=u0_sign)
            self.truth = {}
            self.beta_truth = np.zeros(5)
            self.beta_prior_sd = np.ones(5)
            self.prior_mean, self.prior_sd = np.zeros(8), np.ones(8)
            self.season_edges = np.zeros((1, 2))
            self.phot_times = self.ast_times = np.zeros(2)

        def fit_map(self, starts):
            return optimizer_result(8, objective=1.0), []

        def conditional_beta(self, q):
            return np.zeros(5), np.eye(5)

        def physical(self, q):
            return {"theta_e_mas": 1.0}

        def loglike_jit(self, q):
            return -1.0

    monkeypatch.setattr(roman, "RomanJointFit", Problem)
    monkeypatch.setattr(roman, "save_synthetic", lambda *args: None)
    monkeypatch.setattr(
        roman,
        "sample_branch",
        lambda *args: {
            "sampling": {
                "converged": args[-1] != failed_branch,
                "divergences": 3 if args[-1] == failed_branch else 0,
            }
        },
    )

    def forbidden(*args):
        raise AssertionError("Failed sampling must stop before producing final predictions")

    monkeypatch.setattr(roman, "save_predictive", forbidden)
    args = SimpleNamespace(
        output=tmp_path,
        seed=19,
        starts=1,
        samples=10,
        branch_threshold=15.0,
        skip_disfavored=False,
        negative_warmup=20,
        negative_target_accept=0.98,
    )
    with pytest.raises(RuntimeError, match=f"{failed_branch} posterior convergence check failed"):
        roman.run(args)
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["branches"][failed_branch]["sampling"] == {
        "converged": False,
        "divergences": 3,
    }
