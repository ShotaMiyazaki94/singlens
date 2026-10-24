"""Compare finite-source algorithms against independent brightness integration.

Run from the repository root, with JAX, SciPy, and Matplotlib installed::

    python -m tools.validation.finite_source_methods --quick
    python -m tools.validation.finite_source_methods
    python -m tools.validation.finite_source_methods --nuts --nuts-only

The optional NumPyro experiment uses two chains and reports diagnostics before
interpreting ESS/s. It is a small synthetic transit, not an inference benchmark
representative of all events. Timings synchronize every JAX call.
"""

import argparse
import csv
import hashlib
import inspect
import json
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.integrate import IntegrationWarning, quad

matplotlib.use("Agg")

from example._common import metadata

OUTPUT = Path(__file__).parent / "outputs"
RADII = (1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0)
COEFFICIENTS = (0.0, 0.5, 1.0)


def reference_basis(u, rho, *, tolerance=2e-11):
    """Independent direct source-brightness integral, without disk layering.

    The lens-centered area element cancels the point-source pole. A radial
    substitution removes the square-root limb endpoint. Uniform brightness
    uses an analytic radial primitive; the pure-mu basis uses adaptive radial
    and angular SciPy quadrature. Far wings use source-centered quadrature,
    whose 64/96-order agreement is independently recorded by the caller.
    """
    z = abs(float(u)) / float(rho)
    if z >= 3:
        return wing_basis(u, rho, order=96)

    def angular(theta, power):
        transverse = z * np.sin(theta)
        delta = np.sqrt(max(0.0, 1 - transverse * transverse))
        center = z * np.cos(theta)
        lo, hi = max(0.0, center - delta), max(0.0, center + delta)
        if hi <= lo:
            return 0.0
        if power == 0:
            p, q = rho * lo, rho * hi
            hp, hq = np.hypot(p, 2), np.hypot(q, 2)
            return 0.5 * (hi - lo) * (hq + p * (q + p) / (hq + hp))
        # x = center + delta*sin(t): sqrt(mu^2) = delta*cos(t).
        lower = np.arcsin(np.clip((lo - center) / delta, -1, 1))

        def radial(t):
            x = center + delta * np.sin(t)
            s = rho * x
            return delta**2 * np.cos(t) ** 2 * (s * s + 2) / np.hypot(s, 2)

        return quad(radial, lower, np.pi / 2, epsabs=tolerance * 0.01, epsrel=tolerance, limit=200)[
            0
        ]

    upper = np.pi if z < 1 else np.pi / 2 if z == 1 else np.arcsin(1 / z)
    values = []
    for power in (0, 1):
        value = quad(
            lambda t: angular(t, power),
            0,
            upper,
            points=[np.pi / 2] if z < 1 else None,
            epsabs=tolerance * 0.01,
            epsrel=tolerance,
            limit=250,
        )[0]
        values.append((power + 2) * value / (np.pi * rho))
    return np.asarray(values)


def wing_basis(u, rho, *, order):
    """Smooth source-centered quadrature, for separation >= 3 source radii."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    theta = (nodes + 1) * np.pi / 4
    r, mu = np.sin(theta), np.cos(theta)
    phi = np.arange(2 * order) * np.pi / order
    s = np.sqrt(u * u + (rho * r[:, None]) ** 2 - 2 * abs(u) * rho * r[:, None] * np.cos(phi))
    magnification = (s * s + 2) / (s * np.hypot(s, 2))
    result = []
    for power in (0, 1):
        w = weights * r * mu ** (power + 1)
        result.append(np.sum(w * magnification.mean(axis=1)) / w.sum())
    return np.asarray(result)


class ReferenceCache:
    """Persist costly independent integrals with an implementation fingerprint."""

    def __init__(self, path):
        self.path = path
        self.signature = hashlib.sha256(
            (inspect.getsource(reference_basis) + inspect.getsource(wing_basis)).encode()
        ).hexdigest()
        saved = json.loads(path.read_text()) if path.exists() else {}
        self.values = saved.get("values", {}) if saved.get("signature") == self.signature else {}
        self.warnings = []

    def basis(self, u, rho):
        key = f"{float(u).hex()}:{float(rho).hex()}"
        if key not in self.values:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", IntegrationWarning)
                self.values[key] = reference_basis(u, rho).tolist()
            self.warnings.extend(str(w.message) for w in caught)
        return np.asarray(self.values[key])

    def A(self, u, rho, a1):
        disk, limb = self.basis(u, rho)
        gamma = 2 * a1 / (3 - a1)
        return (1 - gamma) * disk + gamma * limb

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(dict(signature=self.signature, values=self.values)) + "\n")


def methods():
    from singlens.lensing import LinearLimbDarkened
    from singlens.lensing.concentric import ConcentricDisk
    from singlens.lensing.gould import GouldYoo

    return {
        "fftlog": LinearLimbDarkened(),
        "gould_o0": GouldYoo(n_disk=64, n_limb=32, order=0),
        "gould_o2": GouldYoo(n_disk=64, n_limb=32, order=2),
        "gould_o4": GouldYoo(n_disk=64, n_limb=32, order=4),
        "disk_16_8": ConcentricDisk(n_disk=16, n_limb=8),
        "disk_32_16": ConcentricDisk(n_disk=32, n_limb=16),
        "disk_64_32": ConcentricDisk(n_disk=64, n_limb=32),
        "disk_96_64": ConcentricDisk(n_disk=96, n_limb=64),
        "disk_unsplit_64_32": ConcentricDisk(n_disk=64, n_limb=32, split=False),
    }


def accuracy(models, reference, quick):
    ratios = np.array([0, 0.01, 0.2, 0.7, 0.95, 0.99, 0.999, 1, 1.001, 1.01, 1.05, 1.5, 3, 10, 30])
    if not quick:
        ratios = np.unique(
            np.r_[
                ratios,
                np.linspace(0.1, 0.9, 9),
                1 - np.geomspace(1e-6, 0.1, 13),
                1 + np.geomspace(1e-6, 0.1, 13),
            ]
        )
    rows, checks = [], []
    evaluators = {name: jax.jit(model.A) for name, model in models.items()}
    for rho in RADII:
        # Include fixed physical distances as well as source-relative positions.
        positions = np.unique(np.r_[ratios * rho, np.maximum([0.03, 0.1, 1, 10], 3 * rho)])
        bases = np.asarray([reference.basis(u, rho) for u in positions])
        reference.save()
        for u in (
            0,
            0.7 * rho,
            0.999 * rho,
            0.999999 * rho,
            rho,
            1.000001 * rho,
            1.001 * rho,
            3 * rho,
            30 * rho,
        ):
            refined = (
                reference_basis(u, rho, tolerance=2e-13)
                if u < 3 * rho
                else wing_basis(u, rho, order=128)
            )
            checks.append(
                dict(
                    rho=rho,
                    u=float(u),
                    max_relative_change=float(
                        np.max(np.abs(refined / reference.basis(u, rho) - 1))
                    ),
                )
            )
        for a1 in COEFFICIENTS:
            gamma = 2 * a1 / (3 - a1)
            truth = (1 - gamma) * bases[:, 0] + gamma * bases[:, 1]
            for name, evaluate in evaluators.items():
                values = np.asarray(evaluate(jnp.asarray(positions), rho, a1=a1))
                for u, actual, expected in zip(positions, values, truth):
                    rows.append(
                        dict(
                            method=name,
                            rho=rho,
                            a1=a1,
                            u=float(u),
                            ratio=float(u / rho),
                            value=float(actual),
                            reference=float(expected),
                            relative_error=float(actual / expected - 1),
                            finite=bool(np.isfinite(actual)),
                        )
                    )
        print(f"accuracy: rho={rho:g}, {len(positions)} positions", flush=True)
    return rows, checks


def finite_difference(reference, u, rho, a1, parameter, step):
    values = []
    for offset in (-2, -1, 1, 2):
        p = dict(u=u, rho=rho, a1=a1)
        p[parameter] += offset * step
        values.append(reference.A(**p))
    return (values[0] - 8 * values[1] + 8 * values[2] - values[3]) / (12 * step)


def gradients(models, reference, quick):
    ratios = [0, 0.3, 0.99, 1.01, 3] if quick else [0, 0.1, 0.7, 0.99, 0.9999, 1.0001, 1.01, 3, 10]
    evaluators = {
        name: jax.jit(
            jax.value_and_grad(lambda u, r, a, m=model: m.A(u, r, a1=a)[0], argnums=(0, 1, 2))
        )
        for name, model in models.items()
    }
    rows = []
    for rho in RADII:
        for ratio in ratios:
            u = ratio * rho
            for a1 in (0.5,):
                mag = reference.A(u, rho, a1)
                expected, uncertainties = [], []
                for parameter in ("u", "rho"):
                    if parameter == "u" and u == 0:
                        expected.append(0.0)
                        uncertainties.append(0.0)
                        continue
                    # Do not cross the limb singularity with a difference stencil.
                    step = min(1e-4 * rho, abs(u - rho) / 32)
                    coarse = finite_difference(reference, u, rho, a1, parameter, step)
                    fine = finite_difference(reference, u, rho, a1, parameter, step / 2)
                    expected.append(fine + (fine - coarse) / 15)
                    uncertainties.append(abs(fine - coarse) / 15)
                disk, limb = reference.basis(u, rho)
                expected.append(6 * (limb - disk) / (3 - a1) ** 2)
                uncertainties.append(0.0)
                for name, evaluate in evaluators.items():
                    value, actual = evaluate(float(u), float(rho), float(a1))
                    in_domain = bool(np.isfinite(np.asarray(value)))
                    actual = np.asarray(actual)
                    for index, parameter in enumerate(("u", "rho", "a1")):
                        scale = rho if parameter != "a1" else 1.0
                        rows.append(
                            dict(
                                method=name,
                                rho=rho,
                                ratio=ratio,
                                a1=a1,
                                parameter=parameter,
                                finite_magnification=in_domain,
                                value=float(actual[index]),
                                reference=float(expected[index]),
                                scaled_absolute_error=float(
                                    abs(actual[index] - expected[index]) * scale / mag
                                )
                                if in_domain
                                else np.nan,
                                reference_scaled_uncertainty=float(
                                    uncertainties[index] * scale / mag
                                ),
                                relative_error=(
                                    float(actual[index] / expected[index] - 1)
                                    if in_domain and abs(expected[index]) * scale > 1e-10 * mag
                                    else None
                                ),
                            )
                        )
        reference.save()
        print(f"gradients: rho={rho:g}", flush=True)
    return rows


def synchronize(value):
    return jax.tree.map(lambda x: x.block_until_ready(), value)


def timed_compilation_and_calls(function, args, *, repeats):
    start = time.perf_counter()
    compiled = jax.jit(function).lower(*args).compile()
    compilation = time.perf_counter() - start
    synchronize(compiled(*args))
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        synchronize(compiled(*args))
        durations.append(time.perf_counter() - start)
    return dict(
        compilation_seconds=compilation,
        warm_median_seconds=float(np.median(durations)),
        warm_p10_seconds=float(np.quantile(durations, 0.1)),
        warm_p90_seconds=float(np.quantile(durations, 0.9)),
        repeats=repeats,
    )


def runtimes(models, quick):
    rows = []
    for name, model in models.items():
        for size in (1, 64, 512):
            ratios = np.linspace(0.2, 1.8, size)
            for rho in (0.01, 1.0):
                args = (jnp.asarray(ratios * rho), jnp.asarray(rho), jnp.asarray(0.5))

                def function(u, r, a):
                    return model.A(u, r, a1=a)

                def objective(u, r, a):
                    return jnp.sum(jnp.log(model.A(u, r, a1=a)))

                for operation, call in (
                    ("A", function),
                    ("value_and_gradient", jax.value_and_grad(objective, argnums=(0, 1, 2))),
                ):
                    fraction = float(np.mean(np.isfinite(np.asarray(function(*args)))))
                    row = dict(
                        method=name,
                        batch_size=size,
                        rho=rho,
                        operation=operation,
                        finite_output_fraction=fraction,
                    )
                    row.update(timed_compilation_and_calls(call, args, repeats=10 if quick else 40))
                    rows.append(row)
            print(f"runtime: {name}, batch={size}", flush=True)
        # Bound the compilation cache when many large quadrature variants run.
        jax.clear_caches()
    return rows


def nuts(models, reference, args, checkpoint=None):
    import numpyro
    import numpyro.distributions as dist
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    t = np.linspace(-0.18, 0.18, args.nuts_epochs)
    truth = dict(rho=args.nuts_rho, u0=0.4 * args.nuts_rho, a1=0.5, t0=0.0)
    separation = np.hypot(t, truth["u0"])
    mean = np.asarray([reference.A(u, truth["rho"], truth["a1"]) for u in separation])
    noise = 0.003 * mean
    observed = mean + np.random.default_rng(args.seed).normal(size=len(t)) * noise
    reference.save()
    results = []
    for name, evaluator in models.items():
        print(f"NUTS: {name}, 2 chains, {args.warmup} warmup + {args.samples} samples", flush=True)

        def physical(q):
            return (
                truth["rho"] * jnp.exp(0.03 * q[..., 0]),
                truth["u0"] * jnp.exp(0.05 * q[..., 1]),
                jax.nn.sigmoid(0.25 * q[..., 2]),
                0.001 * q[..., 3],
            )

        def model():
            q = numpyro.sample("q", dist.Normal(jnp.zeros(4), 2).to_event(1))
            rho, u0, a1, t0 = physical(q)
            prediction = evaluator.A(jnp.hypot(jnp.asarray(t) - t0, u0), rho, a1=a1)
            numpyro.sample(
                "flux", dist.Normal(prediction, jnp.asarray(noise)), obs=jnp.asarray(observed)
            )

        sampler = MCMC(
            NUTS(model, dense_mass=True, target_accept_prob=0.9, max_tree_depth=9),
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=2,
            chain_method="sequential",
            progress_bar=False,
        )
        key_warm, key_sample = jax.random.split(jax.random.PRNGKey(args.seed))
        init = {"q": jnp.asarray(np.random.default_rng(args.seed + 1).normal(0, 0.1, (2, 4)))}
        start = time.perf_counter()
        sampler.warmup(key_warm, init_params=init)
        synchronize(sampler.post_warmup_state)
        warm_seconds = time.perf_counter() - start
        start = time.perf_counter()
        sampler.run(key_sample, extra_fields=("diverging", "num_steps", "accept_prob"))
        samples = np.asarray(sampler.get_samples(group_by_chain=True)["q"])
        sample_seconds = time.perf_counter() - start
        extras = sampler.get_extra_fields(group_by_chain=True)
        transformed = physical(jnp.asarray(samples))
        posterior = {k: np.asarray(v) for k, v in zip(("rho", "u0", "a1", "t0"), transformed)}
        diagnostics = summary(posterior, group_by_chain=True)
        metrics = {
            k: {field: float(v[field]) for field in ("mean", "std", "n_eff", "r_hat")}
            for k, v in diagnostics.items()
        }
        min_ess = min(v["n_eff"] for v in metrics.values())
        max_rhat = max(v["r_hat"] for v in metrics.values())
        divergences = int(np.sum(extras["diverging"]))
        saturated = int(np.sum(np.asarray(extras["num_steps"]) >= 2**9 - 1))
        results.append(
            dict(
                method=name,
                warmup_and_compile_seconds=warm_seconds,
                sampling_seconds_including_sampling_compile=sample_seconds,
                total_seconds=warm_seconds + sample_seconds,
                min_ess=min_ess,
                max_rhat=max_rhat,
                ess_per_sampling_second=min_ess / sample_seconds,
                ess_per_total_second=min_ess / (warm_seconds + sample_seconds),
                divergences=divergences,
                tree_depth_saturated=saturated,
                mean_leapfrog_steps=float(np.mean(extras["num_steps"])),
                mean_acceptance=float(np.mean(extras["accept_prob"])),
                posterior=metrics,
                diagnostics_pass=bool(
                    divergences == 0 and saturated == 0 and max_rhat < 1.05 and min_ess >= 100
                ),
                truth_max_relative_model_error=float(
                    np.max(
                        np.abs(
                            np.asarray(
                                evaluator.A(jnp.asarray(separation), truth["rho"], a1=truth["a1"])
                            )
                            / mean
                            - 1
                        )
                    )
                ),
            )
        )
        print(
            f"NUTS: {name}, min ESS={min_ess:.0f}, max Rhat={max_rhat:.3f}, "
            f"ESS/s={min_ess / sample_seconds:.2f}",
            flush=True,
        )
        if checkpoint is not None:
            checkpoint(
                dict(
                    results=results,
                    incomplete=True,
                    seed=args.seed,
                    chains=2,
                    warmup_per_chain=args.warmup,
                    samples_per_chain=args.samples,
                )
            )
        jax.clear_caches()
    return dict(
        seed=args.seed,
        chains=2,
        warmup_per_chain=args.warmup,
        samples_per_chain=args.samples,
        observations=args.nuts_epochs,
        truth=truth,
        fixed_parameters="Einstein crossing time=1, source flux=1, blend flux=0",
        priors="q~Normal(0,2): rho=rho_true*exp(.03q0), u0=u0_true*exp(.05q1), "
        "a1=sigmoid(.25q2), t0=.001q3",
        relative_flux_noise=0.003,
        scope="Exploratory single transit and one fixed noise realization; use diagnostics "
        "and model error alongside ESS/s. Sampling time includes sampling executable "
        "compilation; total time also includes adaptation and its compilation.",
        results=results,
    )


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_results(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")
    for key in ("accuracy", "gradients", "runtime"):
        if report.get(key):
            with output.with_name(output.stem + "_" + key + ".csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(report[key][0]))
                writer.writeheader()
                writer.writerows(report[key])


def plot_results(report, output):
    """Compare one representative configuration per numerical method."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    representatives = (
        ("fftlog", "FFTLog", "#247BA0"),
        ("gould_o0", "Gould–Yoo", "#8172B2"),
        ("disk_64_32", "Direct quadrature", "#E07A5F"),
        ("basis_table", "Basis table", "#3D8067"),
    )
    for name, label, color in representatives:
        if name not in report["methods"]:
            continue
        subset = [r for r in report["accuracy"] if r["method"] == name and r["ratio"] <= 3]
        values = [
            max(
                (abs(r["relative_error"]) for r in subset if r["rho"] == rho and r["finite"]),
                default=np.nan,
            )
            for rho in RADII
        ]
        axes[0].loglog(RADII, np.maximum(values, 1e-15), "o-", label=label, color=color)
        subset = [r for r in report["gradients"] if r["method"] == name]
        values = [
            max(
                (
                    r["scaled_absolute_error"]
                    for r in subset
                    if r["rho"] == rho
                    and r["scaled_absolute_error"] is not None
                    and np.isfinite(r["scaled_absolute_error"])
                ),
                default=np.nan,
            )
            for rho in RADII
        ]
        axes[1].loglog(RADII, np.maximum(values, 1e-15), "o-", color=color)
        subset = sorted(
            (
                r
                for r in report["runtime"]
                if r["method"] == name
                and r["rho"] == 0.01
                and r["operation"] == "value_and_gradient"
                and r["finite_output_fraction"] == 1
            ),
            key=lambda r: r["batch_size"],
        )
        axes[2].loglog(
            [r["batch_size"] for r in subset],
            [1e3 * r["warm_median_seconds"] for r in subset],
            "o-",
            color=color,
        )
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].set(xlabel="Source radius rho", ylabel="Maximum relative A error (u/rho <= 3)")
    axes[1].set(xlabel="Source radius rho", ylabel="Maximum scaled gradient error")
    axes[2].set(xlabel="Epochs (rho=0.01, shared radius)", ylabel="Warm value + gradient (ms)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", fontsize=10)
    fig.suptitle("Photometric accuracy and runtime")
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def summarize(report):
    """Keep unsupported-domain outputs distinct from errors on valid outputs."""
    result = []
    for name in report["methods"]:
        for rho in RADII:
            selected = [
                r for r in report.get("accuracy", []) if r["method"] == name and r["rho"] == rho
            ]
            finite = [r for r in selected if r["finite"]]
            grad = [
                r for r in report.get("gradients", []) if r["method"] == name and r["rho"] == rho
            ]
            finite_grad = [
                r
                for r in grad
                if r["scaled_absolute_error"] is not None
                and np.isfinite(r["scaled_absolute_error"])
            ]
            result.append(
                dict(
                    method=name,
                    rho=rho,
                    samples=len(selected),
                    valid_samples=len(finite),
                    nonfinite_samples=len(selected) - len(finite),
                    max_relative_error=max(
                        (abs(r["relative_error"]) for r in finite), default=None
                    ),
                    contact_max_relative_error=max(
                        (abs(r["relative_error"]) for r in finite if r["ratio"] <= 1.1),
                        default=None,
                    ),
                    transit_max_relative_error=max(
                        (abs(r["relative_error"]) for r in finite if r["ratio"] <= 3), default=None
                    ),
                    max_gradient_scaled_error=max(
                        (r["scaled_absolute_error"] for r in finite_grad), default=None
                    ),
                    nonfinite_gradient_samples=len(grad) - len(finite_grad),
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick", action="store_true", help="Sparse accuracy/gradient grids, fewer timing repeats"
    )
    parser.add_argument("--methods", nargs="+", help="Method names to include")
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=("accuracy", "gradients", "runtime"),
        default=["accuracy", "gradients", "runtime"],
    )
    parser.add_argument(
        "--with-table",
        action="store_true",
        help="Build and include the default offline basis table",
    )
    parser.add_argument("--table-radii", type=int, default=96)
    parser.add_argument(
        "--table-refinement",
        type=int,
        default=1,
        help="Subdivide every contact-grid interval by this factor",
    )
    parser.add_argument(
        "--nuts", action="store_true", help="Also run the exploratory two-chain NumPyro experiment"
    )
    parser.add_argument(
        "--nuts-only", action="store_true", help="Skip the accuracy/gradient/runtime grid"
    )
    parser.add_argument("--nuts-methods", nargs="+", default=["fftlog", "gould_o4", "disk_64_32"])
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--nuts-epochs", type=int, default=40)
    parser.add_argument("--nuts-rho", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=260714)
    parser.add_argument("--output", type=Path, default=OUTPUT / "finite_source_methods.json")
    args = parser.parse_args()
    if args.nuts_only:
        args.nuts = True
    if args.samples < 100 or args.warmup < 100:
        parser.error("Use at least 100 warmup and 100 posterior samples per chain.")
    if args.table_radii < 4 or args.table_refinement < 1:
        parser.error("Use at least four table radii and a positive table refinement factor.")
    jax.config.update("jax_enable_x64", True)
    reference = ReferenceCache(OUTPUT / "finite_source_reference.json")
    available = methods()
    table_metadata = None
    if args.with_table:
        from singlens.lensing.basis_table import BasisTable, contact_grid
        from singlens.lensing.concentric import ConcentricDisk

        start = time.perf_counter()
        knots = contact_grid()
        fractions = np.arange(args.table_refinement) / args.table_refinement
        ratios = np.r_[(knots[:-1, None] + np.diff(knots)[:, None] * fractions).ravel(), knots[-1]]
        table = BasisTable.build(
            ConcentricDisk(n_disk=64, n_limb=32),
            rho_grid=np.geomspace(1e-5, 10, args.table_radii),
            z_grid=ratios,
        )
        build_seconds = time.perf_counter() - start
        table_path = args.output.with_name(args.output.stem + "_table.npz")
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.save(table_path)
        available["basis_table"] = table
        table_metadata = dict(
            build_seconds_including_compile=build_seconds,
            radii=len(table.rho_grid),
            ratios=len(table.z_grid),
            generator="ConcentricDisk(n_disk=64,n_limb=32,split=True)",
            file=table_path.name,
            compressed_bytes=table_path.stat().st_size,
            coefficient_bytes=int(table._coefficients.nbytes),
        )
        print(f"table: built in {build_seconds:.2f} s", flush=True)
    selected = available if args.methods is None else {k: available[k] for k in args.methods}
    report = dict(
        **metadata(),
        methods=list(selected),
        quick=args.quick,
        table=table_metadata,
        reference="Independent direct lens-centered brightness integration; "
        "source-centered Gauss/trapezoid quadrature in far wings",
        reference_signature=reference.signature,
        scope="Finite sampled grids, not uniform error bounds. rho is shared per batch. "
        "Gould expansions may be outside their convergence domain at large rho/u. "
        "Gradient error is |AD-reference|*rho/A for u and rho, |AD-reference|/A "
        "for a1; exact limb-contact u/rho gradients are excluded because of the "
        "uniform-brightness logarithmic singularity.",
    )
    if not args.nuts_only:
        if "accuracy" in args.sections:
            report["accuracy"], report["reference_checks"] = accuracy(
                selected, reference, args.quick
            )
            write_results(report, args.output)
        if "gradients" in args.sections:
            report["gradients"] = gradients(selected, reference, args.quick)
            write_results(report, args.output)
        if "runtime" in args.sections:
            report["runtime"] = runtimes(selected, args.quick)
            write_results(report, args.output)
        report["summary"] = summarize(report)
        write_results(report, args.output)
        if all(key in report for key in ("accuracy", "gradients", "runtime")):
            plot_results(report, args.output)
    if args.nuts:

        def checkpoint(value):
            report["nuts"] = value
            write_results(report, args.output)

        report["nuts"] = nuts(
            {k: available[k] for k in args.nuts_methods}, reference, args, checkpoint=checkpoint
        )
    report["reference_integration_warnings"] = sorted(set(reference.warnings))
    write_results(report, args.output)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
