"""Benchmark NUTS with timed sampling blocks and an ESS stopping criterion."""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

from example._cpu import cpu_seconds


def _sampler_statistics(fields):
    """Preserve failed draws in strict JSON, marking nonfinite entries as null."""
    return {
        "axes": ["chain", "draw"],
        "warmup_included": False,
        "nonfinite_counts": {
            name: int((~np.isfinite(values)).sum()) for name, values in fields.items()
        },
        "values": {
            name: np.where(np.isfinite(values), values.astype(object), None).tolist()
            for name, values in fields.items()
        },
    }


def sample_nuts_until(
    objective,
    initial,
    unpack,
    *,
    warmup,
    samples,
    seed,
    max_samples,
    check,
    chains=2,
    chain_method="sequential",
):
    """Independent NUTS chains; diagnostics run outside timed sampling.

    The affine transformation has a constant Jacobian. The objective and
    unpacking function keep priors and summaries in their original coordinates.
    ``samples`` is the minimum per chain; ``check`` may extend it to max_samples.
    """
    if chains < 2 or chain_method not in {"sequential", "parallel", "vectorized"}:
        raise ValueError("Require at least two chains and a supported chain method")
    if chain_method == "parallel" and jax.local_device_count() < chains:
        raise ValueError(
            "Set JAX_NUM_CPU_DEVICES to at least the chain count before starting Python"
        )
    if not 0 < samples <= max_samples:
        raise ValueError("Require 0 < samples <= max_samples.")
    center = jnp.asarray(initial.x)
    transform = jnp.asarray(np.linalg.cholesky(initial.hess_inv))
    target_accept_prob = 0.95
    kernel = NUTS(
        potential_fn=lambda z: objective(center + transform @ z),
        dense_mass=True,
        target_accept_prob=target_accept_prob,
    )
    sampler = MCMC(
        kernel,
        num_warmup=warmup,
        num_samples=min(100, samples),
        num_chains=chains,
        chain_method=chain_method,
        progress_bar=False,
    )
    start, cpu_start = perf_counter(), cpu_seconds()
    init_key, sample_key = jax.random.split(jax.random.key(seed))
    positions = 0.1 * jax.random.normal(init_key, (chains, len(center)))
    sampler.warmup(sample_key, init_params=positions)
    jax.block_until_ready(sampler.last_state)
    warmup_seconds = perf_counter() - start
    warmup_cpu = cpu_seconds() - cpu_start
    print(f"NUTS: {chains} {chain_method} chains warmed up", flush=True)
    blocks, extra, history = [], [], []
    for begin in range(0, max_samples, 100):
        count = min(100, max_samples - begin)
        sampler.num_samples = count
        sampler.post_warmup_state = sampler.last_state
        sampler.run(
            sampler.last_state.rng_key,
            extra_fields=("diverging", "num_steps", "accept_prob", "potential_energy"),
        )
        jax.block_until_ready(sampler.last_state)
        blocks.append(sampler.get_samples(group_by_chain=True))
        extra.append(sampler.get_extra_fields(group_by_chain=True))
        elapsed, cpu_elapsed = perf_counter() - start, cpu_seconds() - cpu_start
        check_start, check_cpu = perf_counter(), cpu_seconds()
        theta = center + jnp.concatenate(blocks, axis=1) @ transform.T
        draws = jax.tree.map(np.asarray, jax.vmap(jax.vmap(unpack))(theta))
        diagnostics = summary(draws, group_by_chain=True)
        fields = {
            name: np.concatenate([block[name] for block in extra], axis=1) for name in extra[0]
        }
        status = check(draws, diagnostics, fields)
        if not all(np.isfinite(values).all() for values in fields.values()):
            status.update(done=False, irrecoverable=True, nonfinite_sampler_statistics=True)
        history.append(
            dict(
                draws_per_chain=begin + count,
                elapsed_seconds=elapsed,
                elapsed_cpu_seconds=cpu_elapsed,
                **status,
            )
        )
        print(f"NUTS: {begin + count} draws/chain, elapsed={elapsed:.1f}s", flush=True)
        start += perf_counter() - check_start
        cpu_start += cpu_seconds() - check_cpu
        if status.get("irrecoverable", False):
            break
        if begin + count >= samples and status.get("done", False):
            break
    report = {
        "warmup_per_chain": warmup,
        "draws_per_chain": begin + count,
        "chains": chains,
        "chain_method": chain_method,
        "seed": seed,
        "target_accept_prob": target_accept_prob,
        "elapsed_seconds_including_compile": elapsed,
        "cpu_seconds_including_compile": cpu_elapsed,
        "warmup_cpu_seconds_including_compile": warmup_cpu,
        "warmup_seconds_including_compile": warmup_seconds,
        "sampling_history": history,
        "sampler_statistics": _sampler_statistics(fields),
        "final_step_size": np.asarray(sampler.last_state.adapt_state.step_size).tolist(),
        "sampler_statistics_scope": "Production states in affine sampling coordinates. Potential energy is the supplied negative log posterior, not total Hamiltonian energy. Step sizes are per-chain final adapted values.",
        "divergences": int(fields["diverging"].sum()),
        "max_r_hat": float(max(item["r_hat"] for item in diagnostics.values())),
        "min_effective_sample_size": float(min(item["n_eff"] for item in diagnostics.values())),
        "mean_accept_prob": (
            float(fields["accept_prob"].mean())
            if np.isfinite(fields["accept_prob"]).all()
            else None
        ),
        "production_draws": int(fields["num_steps"].size),
        "production_leapfrog_steps": int(fields["num_steps"].sum()),
        "mean_num_steps": float(fields["num_steps"].mean()),
        "max_num_steps": int(fields["num_steps"].max()),
        "tree_depth_saturations": int((fields["num_steps"] >= 1023).sum()),
        "parameters": {
            name: {k: float(v) for k, v in values.items()} for name, values in diagnostics.items()
        },
        "stopping_check": status,
    }
    return draws, report
