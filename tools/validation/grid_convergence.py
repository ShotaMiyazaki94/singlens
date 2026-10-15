"""Check FFT refinement against the saved Roman/GP posterior likelihood."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import singlens as sl
from example._common import metadata, save_json
from example.inference.code.inference import OUTPUT as INFERENCE_OUTPUT
from example.inference.code.inference import (
    data_hash,
    event_parameters,
    gp_log_probability,
    load_injection,
)

OUTPUT = Path(__file__).parent / "outputs"


def main():
    jax.config.update("jax_enable_x64", True)
    model, data = load_injection()
    with np.load(INFERENCE_OUTPUT / "posterior.npz") as archive:
        indices = np.linspace(0, archive["t0"].size - 1, 32, dtype=int)
        samples = {k: jnp.asarray(v.reshape(-1)[indices]) for k, v in archive.items()}
    finer = sl.Model(
        source=model.source,
        fft=replace(model.fft, n_fft=2 * model.fft.n_fft),
        parallax=model.parallax,
    )

    @jax.jit
    def compare(params):
        original = model.magnification(data.times, event_parameters(params))
        refined = finer.magnification(data.times, event_parameters(params))
        difference = params["f_s"] * (original - refined) / data.error
        delta_nll = gp_log_probability(
            original, params, data.times, data.flux, data.error
        ) - gp_log_probability(refined, params, data.times, data.flux, data.error)
        return jnp.max(jnp.abs(difference)), jnp.mean(difference**2), delta_nll

    results = np.asarray(
        [compare({k: v[i] for k, v in samples.items()}) for i in range(len(indices))]
    )
    report = {
        **metadata(),
        "data_sha256": data_hash(data),
        "source": model.source,
        "draws_checked": len(indices),
        "n_fft": [model.fft.n_fft, finer.fft.n_fft],
        "max_abs_flux_difference_over_error": float(results[:, 0].max()),
        "rms_flux_difference_over_error": float(np.sqrt(results[:, 1].mean())),
        "delta_gp_nll_median": float(np.median(results[:, 2])),
        "delta_gp_nll_range": [float(results[:, 2].min()), float(results[:, 2].max())],
        "delta_convention": f"NLL({finer.fft.n_fft}) - NLL({model.fft.n_fft})",
    }
    save_json(report, "grid_convergence", OUTPUT)
    print(report)


if __name__ == "__main__":
    main()
