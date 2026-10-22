"""Benchmark NUTS with timed sampling blocks and an ESS stopping criterion."""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS


def sample_nuts_until(objective, initial, unpack, *, warmup, samples, seed, max_samples, check):
    """Two sequential chains; convergence checks run outside timed sampling.

    The affine transformation has a constant Jacobian. The objective and
    unpacking function keep priors and summaries in their original coordinates.
    ``samples`` is the minimum per chain; ``check`` may extend it to max_samples.
    """
    if not 0 < samples <= max_samples:
        raise ValueError("Require 0 < samples <= max_samples.")
    center = jnp.asarray(initial.x)
    transform = jnp.asarray(np.linalg.cholesky(initial.hess_inv))
    kernel = NUTS(
        potential_fn=lambda z: objective(center + transform @ z),
        dense_mass=True,
        target_accept_prob=0.9,
    )
    samplers = [
        MCMC(kernel, num_warmup=warmup, num_samples=min(100, samples), progress_bar=False)
        for _ in range(2)
    ]
    start = perf_counter()
    init_key, sample_key = jax.random.split(jax.random.key(seed))
    positions = 0.1 * jax.random.normal(init_key, (2, len(center)))
    for i, (sampler, key) in enumerate(zip(samplers, jax.random.split(sample_key, 2))):
        sampler.warmup(key, init_params=positions[i])
        jax.block_until_ready(sampler.last_state)
        print(f"NUTS: chain {i + 1} warmup complete", flush=True)
    warmup_seconds = perf_counter() - start
    blocks, extra, history = [[], []], [[], []], []
    for begin in range(0, max_samples, 100):
        count = min(100, max_samples - begin)
        for i, sampler in enumerate(samplers):
            sampler.num_samples = count
            sampler.post_warmup_state = sampler.last_state
            sampler.run(
                sampler.last_state.rng_key, extra_fields=("diverging", "num_steps", "accept_prob")
            )
            jax.block_until_ready(sampler.last_state)
            blocks[i].append(sampler.get_samples())
            extra[i].append(sampler.get_extra_fields())
        elapsed = perf_counter() - start
        check_start = perf_counter()
        theta = center + jnp.stack([jnp.concatenate(chain) for chain in blocks]) @ transform.T
        draws = jax.tree.map(np.asarray, jax.vmap(jax.vmap(unpack))(theta))
        diagnostics = summary(draws, group_by_chain=True)
        fields = {
            name: np.stack([np.concatenate([block[name] for block in chain]) for chain in extra])
            for name in extra[0][0]
        }
        status = check(draws, diagnostics, fields)
        history.append(dict(draws_per_chain=begin + count, elapsed_seconds=elapsed, **status))
        print(f"NUTS: {begin + count} draws/chain, elapsed={elapsed:.1f}s", flush=True)
        start += perf_counter() - check_start
        if begin + count >= samples and status.get("done", False):
            break
    report = {
        "warmup_per_chain": warmup,
        "draws_per_chain": begin + count,
        "chains": 2,
        "seed": seed,
        "elapsed_seconds_including_compile": elapsed,
        "warmup_seconds_including_compile": warmup_seconds,
        "sampling_history": history,
        "divergences": int(fields["diverging"].sum()),
        "max_r_hat": float(max(item["r_hat"] for item in diagnostics.values())),
        "min_effective_sample_size": float(min(item["n_eff"] for item in diagnostics.values())),
        "mean_accept_prob": float(fields["accept_prob"].mean()),
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
