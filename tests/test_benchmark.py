"""Guard the parameter/profile conventions used in external speed comparisons."""

import numpy as np
import pytest

pytestmark = pytest.mark.reference


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
    for curve in curves:
        assert curve.shape == times.shape
        assert np.isfinite(curve).all()
        np.testing.assert_allclose(curve, curve[::-1], rtol=1e-12)
        np.testing.assert_allclose(curve, curves[1], rtol=1e-11 if source == "point" else 5e-3)
    if source == "linear":
        assert results[-1]["gamma"] == 2 * EVENT["a1"] / (3 - EVENT["a1"])
        uniform = evaluators("uniform", times, vbm_tolerances=(1e-6,))
        for curve, item in zip(curves, uniform):
            assert curve[3] > np.asarray(item["call"]())[3]


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
def test_parallax_backend_mapping_and_common_white_likelihood(vbm_interface):
    """VBM must see identical separations, likelihood, and priors."""
    pytest.importorskip("emcee")
    pytest.importorskip("VBMicrolensing")
    pytest.importorskip("tinygp")
    from importlib import import_module

    import jax
    import jax.numpy as jnp

    import singlens as sl
    from example.inference.code.inference import Injection

    benchmark = import_module("example.compare-single-vbml.code.inference_speed")
    recovery = import_module("example.compare-single-vbml.code.experiment")
    model = sl.Model(source="linear", parallax=sl.AnnualParallax(**recovery.PARALLAX))
    times = jnp.array([-30.0, -1.0, -0.17, 0.2, 0.36, 2.0, 30.0])
    theta = jnp.asarray(recovery.PRIOR_LOC).at[0].set(0.2).at[6:8].set(jnp.array([0.3, -0.2]))
    p = recovery.unpack(theta)
    flux = (
        p["f_s"] * model.magnification(times, recovery.event_parameters(p))
        + p["f_b"]
        + jnp.array([0.01, -0.02, 0.03, 0.01, 0.0, -0.03, 0.02])
    )
    data = Injection(times, flux, jnp.full_like(times, 0.02), p, flux, jnp.zeros_like(times))
    posterior = benchmark.Posterior(model, data, "VBM", vbm_interface=vbm_interface)
    u = posterior.separations(theta)[0]
    np.testing.assert_allclose(
        u, np.abs(model.trajectory(times, recovery.event_parameters(p))), atol=1e-13
    )
    # Different coefficients within one walker batch must not share VBM state.
    theta_batch = np.repeat(np.asarray(theta)[None], 2, axis=0)
    theta_batch[:, 8] = np.log(np.array([0.2, 0.8]) / np.array([0.8, 0.2]))
    curves = posterior.magnifications(theta_batch)
    for coefficient, values in zip((0.2, 0.8), curves):
        posterior.vbm.a1 = coefficient
        expected = [posterior.vbm.ESPLMag2(float(x), float(p["rho"])) for x in u]
        np.testing.assert_allclose(values, expected, rtol=1e-13)
    assert not np.allclose(*curves)
    assert benchmark.check_vbm_interface(posterior, theta_batch)["passed"]
    np.testing.assert_allclose(benchmark.physical(theta_batch)[:, 8], [0.2, 0.8])
    magnification = model.magnification(times, recovery.event_parameters(p))
    common = posterior.log_prob(theta[None], magnification[None])[0]
    original = -jax.jit(recovery.negative_log_posterior, static_argnums=1)(
        theta, model, times, flux, data.error
    )
    np.testing.assert_allclose(common, original, atol=1e-10)


def test_emcee_counts_initial_states_and_cached_production_states(tmp_path, monkeypatch):
    """Record evaluations, including warmup's initial ensemble but not cached states."""
    pytest.importorskip("emcee")
    pytest.importorskip("tinygp")
    from importlib import import_module
    from types import SimpleNamespace

    benchmark = import_module("example.compare-single-vbml.code.inference_speed")
    dimensions = len(benchmark.recovery.NAMES)
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
    from example.inference.code import inference as recovery

    benchmark = import_module("example.compare-single-vbml.code.inference_speed")
    integral = import_module("example.compare-single-vbml.code.accuracy").integral
    model = sl.Model(source="linear", parallax=sl.AnnualParallax(**recovery.PARALLAX))
    times = jnp.array([0.0, 1.0])
    data = recovery.Injection(times, times, jnp.ones(2), {}, times, times)
    engine = benchmark.Posterior(model, data, "VBM").vbm
    u, rho, a1 = 0.005396705070645264, 0.009948808908776117, 0.5116990888449133
    engine.a1 = a1
    np.testing.assert_allclose(engine.ESPLMag2(u, rho), integral(u, rho, a1=a1), rtol=2e-3)


def test_common_ess_on_autoregressive_series():
    pytest.importorskip("emcee")
    pytest.importorskip("tinygp")
    from importlib import import_module

    benchmark = import_module("example.compare-single-vbml.code.inference_speed")
    rng = np.random.default_rng(40)
    series = rng.normal(size=(16000, 8, 1))
    for i in range(1, len(series)):
        series[i] += 0.8 * series[i - 1]
    # AR(1): tau=(1+phi)/(1-phi)=9, independent of marginal normalization.
    np.testing.assert_allclose(benchmark.autocorrelation(series), [9.0], rtol=0.12)


def test_inference_comparison_never_evaluates_gp(monkeypatch):
    """Generate fresh independent noise and evaluate the nine-parameter objective."""
    pytest.importorskip("tinygp")
    from importlib import import_module

    import jax.numpy as jnp

    experiment = import_module("example.compare-single-vbml.code.experiment")

    def unexpected_gp(*args, **kwargs):
        raise AssertionError("The inference comparison must not evaluate a GP")

    monkeypatch.setattr(experiment.stellar, "build_gp", unexpected_gp)
    model, data = experiment.synthetic_event()
    assert set(data.truth) == set(experiment.NAMES) and len(data.truth) == 9
    np.testing.assert_array_equal(data.gp_signal, 0)
    residual = np.asarray((data.flux - data.lens_flux) / data.error)
    assert abs(residual.mean()) < 0.04
    assert 0.96 < residual.std() < 1.04
    value = experiment.negative_log_posterior(
        jnp.asarray(experiment.PRIOR_LOC), model, data.times, data.flux, data.error
    )
    assert np.isfinite(value)
