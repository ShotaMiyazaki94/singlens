"""Guard the parameter/profile conventions used in external speed comparisons."""

import numpy as np
import pytest

pytestmark = pytest.mark.reference
pytest.importorskip("matplotlib")


def inference_benchmark():
    """Skip optional inference checks before importing their plotting/model stack."""
    from importlib import import_module

    for package in ("numpyro", "emcee", "corner", "tinygp"):
        pytest.importorskip(package)
    return import_module("example.compare-single-vbml.code.inference_speed")


@pytest.mark.parametrize("source", ["point", "uniform", "linear"])
def test_native_batch_benchmark_agrees_on_event(source):
    pytest.importorskip("matplotlib")
    pytest.importorskip("VBMicrolensing")
    pytest.importorskip("MulensModel")
    from importlib import import_module

    benchmark = import_module("example.compare-single-vbml.code.benchmark")
    EVENT, evaluators = benchmark.EVENT, benchmark.evaluators

    times = np.array([-0.7, -0.21, -0.1, 0.0, 0.1, 0.21, 0.7])
    results = evaluators(source, times, vbm_tolerances=(1e-6,))
    curves = [np.asarray(item["call"]()) for item in results]
    reference = next(
        curve
        for item, curve in zip(results, curves)
        if item["method"] in ("PSPLLightCurve", "ESPLLightCurve")
    )
    for curve in curves:
        assert curve.shape == times.shape
        assert np.isfinite(curve).all()
        np.testing.assert_allclose(curve, curve[::-1], rtol=1e-12)
        np.testing.assert_allclose(curve, reference, rtol=1e-11 if source == "point" else 5e-3)
    if source == "linear":
        mulens = next(item for item in results if item["implementation"] == "MulensModel")
        assert mulens["gamma"] == 2 * EVENT["a1"] / (3 - EVENT["a1"])
        uniform = evaluators("uniform", times, vbm_tolerances=(1e-6,))
        uniform_by_method = {(item["implementation"], item.get("n_fft")): item for item in uniform}
        for item, curve in zip(results, curves):
            uniform_item = uniform_by_method[item["implementation"], item.get("n_fft")]
            assert curve[3] > np.asarray(uniform_item["call"]())[3]


@pytest.mark.parametrize("source", ["uniform", "linear"])
def test_runtime_comparison_uses_default_table_and_explicit_fft(source):
    """A reported backend must evaluate the corresponding public Model."""
    pytest.importorskip("matplotlib")
    from importlib import import_module

    import singlens as sl

    benchmark = import_module("example.compare-single-vbml.code.benchmark")
    times = np.array([-0.21, 0.0, 0.13, 0.7])
    rows = benchmark.evaluators(source, times, compare=False)
    assert {row["solver"] for row in rows} == {"table", "fft"}
    assert len(rows) == 2 and all(row["primary"] for row in rows)
    event = benchmark.EVENT
    params = sl.Parameters(
        t0=event["t0"],
        u0=event["u0"],
        t_e=event["t_e"],
        rho=event["rho"],
        a1=event["a1"] if source == "linear" else None,
    )
    for row in rows:
        model = sl.Model(source=source, solver=None if row["solver"] == "table" else "fft")
        np.testing.assert_allclose(row["call"](), model.magnification(times, params), rtol=2e-13)
        assert row["solver_metadata"]["solver"] == row["solver"]
        assert row["solver_metadata"]["source"] == source
        assert row["n_fft"] == (None if row["solver"] == "table" else 4096)
    diagnostic = benchmark.evaluators(source, times, compare=False, include_table=False, n_fft=128)
    assert len(diagnostic) == 1
    assert diagnostic[0]["solver"] == "fft" and diagnostic[0]["n_fft"] == 128
    assert not diagnostic[0]["primary"]


def test_solver_provenance_describes_actual_model_and_packaged_artifact():
    pytest.importorskip("matplotlib")
    from dataclasses import asdict
    from hashlib import sha256
    from importlib.resources import files

    import singlens as sl
    from example._common import model_solver_metadata
    from singlens.lensing import FFTConfig

    table = sl.Model(source="linear")
    provenance = model_solver_metadata(table)
    assert provenance["solver"] == "table" and provenance["origin"] == "bundled"
    assert provenance["configuration"]["coefficient_shape"] == list(
        table.solver._coefficients.shape
    )
    assert provenance["configuration"]["rho"] == [1e-6, 10.0]
    assert provenance["configuration"]["u_over_rho"] == [0.0, 1e8]
    assert provenance["configuration"]["dtype"] == str(table.solver._coefficients.dtype)
    resource = files("singlens.data").joinpath(provenance["artifact"]["file"])
    assert provenance["artifact"]["zip_sha256"] == sha256(resource.read_bytes()).hexdigest()
    assert "n_fft" not in provenance["configuration"]
    settings = FFTConfig(n_fft=128)
    fft = model_solver_metadata(sl.Model(source="linear", solver=settings))
    assert fft["solver"] == "fft" and fft["configuration"] == asdict(settings)
    assert "artifact" not in fft
    point = model_solver_metadata(sl.Model(source="point"))
    assert point["solver"] == "point" and point["configuration"] == {}


def test_primary_runtime_methods_exclude_extra_references():
    pytest.importorskip("matplotlib")
    pytest.importorskip("VBMicrolensing")
    pytest.importorskip("MulensModel")
    from importlib import import_module

    benchmark = import_module("example.compare-single-vbml.code.benchmark")
    rows = benchmark.evaluators("linear", np.array([-0.1, 0.1]), vbm_tolerances=(1e-3, 1e-6))
    primary = [row for row in rows if row["primary"]]
    assert len(primary) == 3
    assert {row["implementation"] for row in primary} == {
        "singlens table",
        "singlens FFTLog",
        "VBMicrolensing",
    }
    vbm = next(row for row in primary if row["implementation"] == "VBMicrolensing")
    assert vbm["absolute_tolerance"] == vbm["relative_tolerance"] == 1e-3
    assert not next(row for row in rows if row["implementation"] == "MulensModel")["primary"]


@pytest.mark.parametrize("solver", ["table", "FFTLog"])
def test_accuracy_rejects_nonfinite_singlens_results(solver):
    """A table domain failure must not disappear from an error summary."""
    pytest.importorskip("matplotlib")
    from importlib import import_module

    accuracy = import_module("example.compare-single-vbml.code.accuracy")
    with pytest.raises(RuntimeError, match="Non-finite"):
        accuracy.add_result(
            [],
            [],
            "linear",
            f"singlens {solver}",
            0.01,
            np.array([2.0, np.nan]),
            np.array([2.0, 1.0]),
        )


def test_accuracy_reference_normalization_and_independent_geometry():
    """Check the benchmark reference against a source-centered center integral."""
    from importlib import import_module

    pytest.importorskip("matplotlib")
    from scipy.integrate import quad

    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    for rho in (1e-6, 0.01, 1.0, 10.0):
        np.testing.assert_allclose(integral(0.0, rho), np.hypot(rho, 2.0) / rho, rtol=1e-12)
        contact = 2 * (rho + (1 + rho**2) * np.arctan(rho)) / (np.pi * rho**2)
        np.testing.assert_allclose(integral(rho, rho), contact, rtol=1e-11)
        a1 = 0.5

        def radial(theta):
            x = rho * np.sin(theta)
            return ((1 - a1) + a1 * np.cos(theta)) * (x * x + 2) / np.hypot(x, 2) * np.cos(theta)

        expected = 2 * quad(radial, 0, np.pi / 2, epsabs=1e-12, epsrel=1e-12)[0]
        expected /= rho * (1 - a1 / 3)
        np.testing.assert_allclose(integral(0.0, rho, a1=a1), expected, rtol=1e-10)


def test_reference_near_limb_against_elliptic_small_source_limit():
    from importlib import import_module

    pytest.importorskip("matplotlib")
    from scipy.special import ellipe, ellipk

    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    rho = 1e-6
    for ratio in (0.5, 0.99, 0.995, 1.005, 1.01, 2.0):
        if ratio < 1:
            expected = 4 * ellipe(ratio**2) / (np.pi * rho)
        else:
            m = 1 / ratio**2
            expected = 4 * ratio * (ellipe(m) - (1 - m) * ellipk(m)) / (np.pi * rho)
        np.testing.assert_allclose(integral(ratio * rho, rho), expected, rtol=1e-10)


@pytest.mark.parametrize("vbm_interface", ["array", "python"])
def test_exposure_backend_mapping_and_common_white_likelihood(vbm_interface):
    """VBM must average identical subexposures and use fixed errors and priors."""
    pytest.importorskip("emcee")
    pytest.importorskip("VBMicrolensing")
    pytest.importorskip("tinygp")
    from importlib import import_module

    import jax
    import jax.numpy as jnp

    benchmark = inference_benchmark()
    recovery = import_module("example.compare-single-vbml.code.experiment")
    model = recovery.make_model()
    times = jnp.array([-30.0, -0.1, 0.14, 0.2, 0.26, 0.4, 30.0])
    theta = jnp.asarray(recovery.PRIOR_LOC).at[0].set(0.2)
    p = {**recovery.CONFIG["event"], **recovery.unpack(theta)}
    flux = (
        p["f_s"] * recovery.magnification(model, times, p)
        + p["f_b"]
        + jnp.array([0.01, -0.02, 0.03, 0.01, 0.0, -0.03, 0.02])
    )
    data = recovery.Injection(
        times, flux, jnp.full_like(times, 0.02), p, flux, jnp.zeros_like(times)
    )
    posterior = benchmark.Posterior(model, data, "VBM", vbm_interface=vbm_interface)
    u = posterior.separations(theta)[0]
    np.testing.assert_allclose(
        u,
        np.abs(
            model.trajectory(recovery.exposure_times(times), recovery.event_parameters(p))
        ).ravel(),
        atol=1e-13,
    )
    # Distinct lens radii in one walker batch must remain distinct.
    theta_batch = np.repeat(np.asarray(theta)[None], 2, axis=0)
    theta_batch[:, 3] = np.log([0.4, 0.8])
    curves = posterior.magnifications(theta_batch)
    _, weights = recovery.exposure_rule(recovery.EXPOSURE_ORDER)
    for row, values in zip(theta_batch, curves):
        parameters = recovery.unpack(row)
        posterior.vbm.a1 = recovery.CONFIG["event"]["a1"]
        sub_u = np.abs(
            model.trajectory(recovery.exposure_times(times), recovery.event_parameters(parameters))
        )
        expected = (
            np.array(
                [posterior.vbm.ESPLMag2(float(x), float(parameters["rho"])) for x in sub_u.ravel()]
            ).reshape(sub_u.shape)
            @ weights
        )
        np.testing.assert_allclose(values, expected, rtol=1e-13)
    assert not np.allclose(*curves)
    assert benchmark.check_vbm_interface(posterior, theta_batch)["passed"]
    assert benchmark.physical(theta_batch).shape == (2, 4)
    np.testing.assert_allclose(
        benchmark.transformed(benchmark.physical(theta_batch)), theta_batch, atol=1e-13
    )
    magnification = recovery.magnification(model, times, p)
    common = posterior.log_prob(theta[None], magnification[None])[0]
    original = -jax.jit(recovery.negative_log_posterior, static_argnums=1)(
        theta, model, times, flux, data.error
    )
    np.testing.assert_allclose(common, original, atol=1e-10)
    # A trial light curve is integrated against the same proper Gaussian prior
    # and fixed measured variances, including the likelihood normalization.
    trial = np.asarray(magnification) * 1.03
    conditional = recovery.flux_conditional(trial, data.flux, data.error)
    expected = conditional.log_likelihood + recovery.log_prior(theta)
    np.testing.assert_allclose(
        posterior.log_prob(theta[None], trial[None])[0], expected, atol=1e-10
    )
    table = benchmark.Posterior(model, data, "singlens")
    fft = benchmark.Posterior(recovery.make_model("fft"), data, "singlens")
    for backend in (table, fft):
        np.testing.assert_allclose(
            backend.log_prob(theta[None], trial[None])[0], expected, atol=1e-10
        )
    outside = theta.at[3].set(np.log(11.0))
    assert np.isneginf(posterior.log_prob(outside[None], magnification[None])[0])


def test_emcee_counts_initial_states_and_cached_production_states(tmp_path, monkeypatch):
    """Record evaluations, including warmup's initial ensemble but not cached states."""
    pytest.importorskip("emcee")
    pytest.importorskip("tinygp")
    from types import SimpleNamespace

    benchmark = inference_benchmark()
    dimensions = len(benchmark.recovery.LENS_NAMES)
    initial = SimpleNamespace(x=np.zeros(dimensions), hess_inv=np.eye(dimensions))

    class Posterior:
        backend = "test"

        def __call__(self, theta):
            return -0.5 * np.sum(theta**2, axis=1)

    monkeypatch.setattr(benchmark, "autocorrelation", lambda chains: np.ones(dimensions))
    chains, seconds, info = benchmark.run_emcee(
        Posterior(),
        initial,
        warmup=3,
        chunk=50,
        max_steps=100,
        walkers=20,
        seed=2,
        output=tmp_path,
        target_ess=1,
    )
    counts = info["posterior_evaluations"]
    assert counts["warmup_including_initial"] == 2 * 20 * 4
    assert counts["production"] == 2 * 20 * 100
    assert counts["total"] == 4160
    assert counts["warmup_callback_calls"] == 2 * (1 + 2 * 3)
    assert counts["production_callback_calls"] == 2 * 2 * 100
    result = benchmark.efficiency(
        chains, seconds, initialization_seconds=2, backend_setup_seconds=3
    )
    assert result["total_seconds"] == seconds + 5
    assert result["min_ess_per_second"] == result["min_ess"] / (seconds + 5)


def test_vbm_limb_integration_resolves_disk_point_coincidence():
    """A near equality of disk and point magnification is not a uniform profile."""
    pytest.importorskip("emcee")
    pytest.importorskip("VBMicrolensing")
    pytest.importorskip("tinygp")
    from importlib import import_module

    import jax.numpy as jnp

    import singlens as sl

    benchmark = inference_benchmark()
    recovery = import_module("example.compare-single-vbml.code.experiment")
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    model = sl.Model(source="linear")
    times = jnp.array([0.0, 1.0])
    data = recovery.Injection(times, times, jnp.ones(2), {}, times, times)
    engine = benchmark.Posterior(model, data, "VBM").vbm
    u, rho, a1 = 0.005396705070645264, 0.009948808908776117, 0.5116990888449133
    engine.a1 = a1
    np.testing.assert_allclose(engine.ESPLMag2(u, rho), integral(u, rho, a1=a1), rtol=2e-3)


def test_common_ess_on_autoregressive_series():
    pytest.importorskip("emcee")
    pytest.importorskip("tinygp")

    benchmark = inference_benchmark()
    rng = np.random.default_rng(40)
    series = rng.normal(size=(16000, 8, 1))
    for i in range(1, len(series)):
        series[i] += 0.8 * series[i - 1]
    # AR(1): tau=(1+phi)/(1-phi)=9, independent of marginal normalization.
    np.testing.assert_allclose(benchmark.autocorrelation(series), [9.0], rtol=0.12)


def test_independent_injection_is_identical_for_fitting_solvers(monkeypatch, tmp_path):
    """Data generation must never call either tested solver or change with it."""
    from importlib import import_module

    import singlens as sl
    from singlens.lensing import BasisTable, FFTConfig

    experiment = import_module("example.compare-single-vbml.code.experiment")

    def unexpected_solver(*args, **kwargs):
        raise AssertionError("Injection must not call a tested magnification solver")

    times = np.array([-30, -0.1, 0.1, 0.13, 0.2, 0.27, 0.3, 0.5, 30.0])
    monkeypatch.setattr(experiment, "observation_times", lambda: times)
    monkeypatch.setattr(sl.Model, "magnification", unexpected_solver)
    reference = tmp_path / "independent_reference.npz"
    model, data = experiment.synthetic_event(reference=reference)
    assert reference.exists()
    # A cache hit must retain the validated independent mean without reintegrating.
    accuracy = import_module("example.compare-single-vbml.code.accuracy")
    monkeypatch.setattr(accuracy, "quad", unexpected_solver)
    fft_model, fft_data = experiment.synthetic_event(solver="fft", reference=reference)
    assert isinstance(model.solver, BasisTable)
    assert isinstance(fft_model.solver, FFTConfig) and fft_model.solver.n_fft == 4096
    assert experiment.data_hash(data) == experiment.data_hash(fft_data)
    for name in data._fields:
        if name in ("truth", "reference_checks", "seed"):
            assert getattr(data, name) == getattr(fft_data, name)
        else:
            np.testing.assert_array_equal(getattr(data, name), getattr(fft_data, name))
    reports = [experiment.experiment_metadata(m, data) for m in (model, fft_model)]
    assert reports[0]["data_sha256"] == reports[1]["data_sha256"]
    assert reports[0]["injection_solver"] == reports[1]["injection_solver"]
    assert "Independent" in reports[0]["injection_solver"]
    assert reports[0]["fit_solver"]["solver"] == "table"
    assert reports[1]["fit_solver"]["solver"] == "fft"
    assert "n_fft" not in reports[0] and "fft" not in reports[0]
    assert set(data.truth) == set(experiment.NAMES) and len(data.truth) == 6
    assert model.parallax is None and fft_model.parallax is None
    np.testing.assert_array_equal(data.gp_signal, 0)
    residual = np.asarray((data.flux - data.lens_flux) / data.error)
    np.testing.assert_allclose(
        residual, np.random.default_rng(data.seed).normal(size=len(times)), atol=5e-14
    )
    changed_seed = experiment.synthetic_data(seed=data.seed + 1, reference=reference)
    assert experiment.data_hash(data) != experiment.data_hash(changed_seed)
    for name in ("times", "lens_flux", "error", "gp_signal"):
        np.testing.assert_array_equal(getattr(data, name), getattr(changed_seed, name))
    assert data.reference_checks["injection_refinement_max_flux_error_sigma"] < 1e-3


def test_finite_exposure_objective_gradients_match_directional_difference():
    from importlib import import_module

    import jax
    import jax.numpy as jnp

    experiment = import_module("example.compare-single-vbml.code.experiment")
    benchmark = inference_benchmark()
    model = experiment.make_model()
    params = {name: experiment.CONFIG["event"][name] for name in experiment.NAMES}
    theta = benchmark.transformed(np.array([params[name] for name in experiment.NAMES]))
    # Include the transit, nearby wings, and baseline; avoid exact limb contact.
    times = jnp.asarray([-30, -0.1, 0.1, 0.13, 0.2, 0.27, 0.3, 0.5, 30.0])
    magnification = experiment.reference_magnification(times, params)
    flux = jnp.asarray(
        params["f_s"] * magnification + params["f_b"] + np.linspace(-0.002, 0.003, len(times))
    )
    error = jnp.asarray(
        experiment.photometric_error(
            params["f_s"] * magnification + params["f_b"], experiment.CONFIG
        )
    )

    @jax.jit
    def objective(theta):
        return experiment.negative_log_posterior(theta, model, times, flux, error)

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(theta))
    assert np.isfinite(value) and np.isfinite(gradient).all()
    directions = np.array([[1, -2, 3, 2], [-2, 1, 1, -3]])
    for direction in directions * experiment.PRIOR_SCALE:
        direction /= np.linalg.norm(direction)
        step = 1e-5
        difference = (objective(theta + step * direction) - objective(theta - step * direction)) / (
            2 * step
        )
        np.testing.assert_allclose(gradient @ direction, difference, rtol=5e-4, atol=1e-5)


def test_marginal_target_integrates_the_stated_correlated_flux_prior():
    """Check the new target independently in source/excess-baseline coordinates."""
    from importlib import import_module

    experiment = import_module("example.compare-single-vbml.code.experiment")
    magnification = np.array([1.0, 1.2, 2.3, 1.5, 1.0])
    flux = np.array([1.01, 1.15, 2.04, 1.39, 0.99])
    error = np.array([0.02, 0.03, 0.025, 0.02, 0.015])
    # These are the scientific prior's stated source and total-baseline moments.
    prior_mean = np.array([0.8, 1.0])
    prior_covariance = np.array([[0.0272, 0.002], [0.002, 0.0025]])
    design = np.column_stack((magnification - 1, np.ones_like(magnification)))
    total = np.diag(error**2) + design @ prior_covariance @ design.T
    residual = flux - design @ prior_mean
    expected_logp = -0.5 * (
        residual @ np.linalg.solve(total, residual)
        + np.linalg.slogdet(total)[1]
        + len(flux) * np.log(2 * np.pi)
    )
    gain = prior_covariance @ design.T @ np.linalg.inv(total)
    expected_mean = prior_mean + gain @ residual
    expected_covariance = prior_covariance - gain @ design @ prior_covariance
    to_source_blend = np.array([[1, 0], [-1, 1]])
    result = experiment.flux_conditional(magnification, flux, error)
    np.testing.assert_allclose(result.log_likelihood, expected_logp, atol=1e-10)
    np.testing.assert_allclose(result.mean, to_source_blend @ expected_mean, atol=1e-12)
    np.testing.assert_allclose(
        result.covariance,
        to_source_blend @ expected_covariance @ to_source_blend.T,
        atol=1e-12,
    )
    np.testing.assert_allclose(experiment.log_likelihood(magnification, flux, error), expected_logp)
    assert np.linalg.eigvalsh(result.covariance).min() > 0


def test_only_four_lens_coordinates_are_transformed_and_sampled():
    from importlib import import_module

    experiment = import_module("example.compare-single-vbml.code.experiment")
    benchmark = inference_benchmark()
    assert experiment.LENS_NAMES == ("t0", "u0", "t_e", "rho")
    assert len(experiment.TRANSFORM_NAMES) == len(experiment.PRIOR_LOC) == 4
    physical = np.array([0.2, 0.4, 0.12, 0.6, 0.8, 0.2])
    theta = benchmark.transformed(physical)
    assert theta.shape == (4,)
    np.testing.assert_allclose(theta, [0.2, np.log(0.4 / 0.6), np.log(0.12 * 0.6), np.log(0.6)])
    np.testing.assert_allclose(benchmark.physical(theta), physical[:4])
    assert set(experiment.unpack(theta)) == set(experiment.LENS_NAMES)
    # Posterior flux draws do not become additional sampler coordinates.
    physical[4:] = [-0.1, 1.3]
    np.testing.assert_array_equal(benchmark.transformed(physical), theta)
    expected_prior = -0.5 * np.sum(((theta - experiment.PRIOR_LOC) / experiment.PRIOR_SCALE) ** 2)
    np.testing.assert_allclose(experiment.log_prior(theta), expected_prior)


def test_efficiency_names_only_the_correlated_lens_chain(monkeypatch):

    benchmark = inference_benchmark()
    chains = np.arange(40 * 2 * 4, dtype=float).reshape(40, 2, 4) / 100
    expected_tau = np.array([2.0, 3.0, 4.0, 5.0])

    def autocorrelation(values):
        np.testing.assert_array_equal(values, chains)
        return expected_tau

    monkeypatch.setattr(benchmark, "autocorrelation", autocorrelation)
    result = benchmark.efficiency(
        chains, 7.0, initialization_seconds=2.0, backend_setup_seconds=1.0
    )
    assert result["ess_parameter_names"] == list(benchmark.recovery.LENS_NAMES)
    assert set(result["parameters"]) == set(benchmark.recovery.LENS_NAMES)
    assert result["min_ess"] == 80 / 5
    assert result["min_ess_per_second"] == 80 / 5 / 10
    for name, tau in zip(benchmark.recovery.LENS_NAMES, expected_tau):
        assert result["parameters"][name]["ess"] == 80 / tau
    # Reject reconstructed flux draws before autocorrelation can use their
    # independent conditional noise to inflate effective sample sizes.
    for wrong in (np.zeros((40, 2, 6)), np.zeros((40, 4))):
        with pytest.raises(ValueError, match="four|shape|lens"):
            benchmark.efficiency(wrong, 7.0, initialization_seconds=2.0)


def test_conditional_flux_reconstruction_retains_covariance_and_excludes_ess():
    from types import SimpleNamespace

    benchmark = inference_benchmark()
    experiment = benchmark.recovery
    a, flux, error = np.ones(3), np.ones(3), np.full(3, 0.03)
    calls = []

    class Backend:
        backend = "test"
        data = SimpleNamespace(flux=flux, error=error)

        def magnifications(self, theta):
            calls.append(np.array(theta))
            return np.broadcast_to(a, (len(theta), len(a)))

    lens = np.broadcast_to([0.2, 0.4, 0.12, 0.6], (32, 64, 4))
    posterior = Backend()
    values, info, means, covariances = benchmark.reconstruct_flux(posterior, lens, seed=123)
    assert values.shape == (32, 64, 6)
    np.testing.assert_array_equal(values[..., :4], lens)
    assert means.shape == (32, 64, 2) and covariances.shape == (32, 64, 2, 2)
    assert max(len(block) for block in calls) <= 16
    np.testing.assert_allclose(
        np.concatenate(calls), np.broadcast_to(benchmark.transformed(lens[0, 0]), (2048, 4))
    )
    # With constant A the data constrain only baseline, so source/blend retain
    # nonzero anticorrelated uncertainty. Conditional means alone would fail.
    design = np.column_stack((a, np.ones_like(a)))
    prior = experiment.FLUX_PRIOR_COVARIANCE
    covariance = np.linalg.inv(np.linalg.inv(prior) + design.T @ (design / error[:, None] ** 2))
    np.testing.assert_allclose(
        covariances, np.broadcast_to(covariance, covariances.shape), rtol=1e-11
    )
    drawn = values[..., 4:].reshape(-1, 2)
    np.testing.assert_allclose(np.cov(drawn.T), covariance, rtol=0.12, atol=3e-4)
    np.testing.assert_allclose(drawn.mean(axis=0), means[0, 0], atol=0.015)
    assert info["geometry_states"] == 2048 and info["backend"] == "test"
    from scipy.special import ndtr

    expected_tail = ndtr(-means[0, 0] / np.sqrt(np.diag(covariance)))
    for index, name in enumerate(("f_s", "f_b")):
        np.testing.assert_allclose(
            info["negative_flux_probabilities"][name]["conditional_probability_mean"],
            expected_tail[index],
            rtol=1e-10,
        )
    changed, _, changed_means, changed_covariances = benchmark.reconstruct_flux(
        posterior, lens[:1], seed=456
    )
    np.testing.assert_array_equal(changed[..., :4], lens[:1])
    np.testing.assert_array_equal(changed_means, means[:1])
    np.testing.assert_array_equal(changed_covariances, covariances[:1])
    assert not np.array_equal(changed[..., 4:], values[:1, ..., 4:])
    row = {"parameters": {name: {"ess": 20} for name in experiment.LENS_NAMES}}
    benchmark.add_flux_summary(row, values, info)
    for name in ("f_s", "f_b"):
        assert row["parameters"][name]["sd"] > 0
        assert not {"tau", "ess", "mcse_mean"}.intersection(row["parameters"][name])
    assert all(row["parameters"][name]["ess"] == 20 for name in experiment.LENS_NAMES)
    with pytest.raises(ValueError, match="four"):
        benchmark.reconstruct_flux(posterior, values, seed=1)


def test_exposure_average_has_correct_time_units_and_normalization():
    from importlib import import_module

    experiment = import_module("example.compare-single-vbml.code.experiment")
    times = np.array([[-0.01, 0.0], [0.01, 0.02]])
    truth = experiment.CONFIG["event"]

    class QuadraticCurve:
        def magnification(self, t, parameters):
            return 1 + (t - parameters.t0) ** 2

    averaged = experiment.magnification(QuadraticCurve(), times, truth)
    duration_days = experiment.CONFIG["exposure_seconds"] / 86400
    expected = 1 + (times - truth["t0"]) ** 2 + duration_days**2 / 12
    assert averaged.shape == times.shape
    np.testing.assert_allclose(averaged, expected, rtol=0, atol=5e-16)
