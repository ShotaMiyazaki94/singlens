"""Independent-noise Roman-like experiment for the inference comparison."""

import hashlib
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from example._common import metadata
from example.inference.code import inference as stellar
from example.inference.code.inference import PARALLAX, data_hash, event_parameters

NAMES = ("t0", "u0", "t_e", "rho", "f_s", "f_b", "pi_e_n", "pi_e_e", "a1")
_indices = [stellar.NAMES.index(name) for name in NAMES]
LABELS = tuple(stellar.LABELS[i] for i in _indices)
PRIOR_LOC = stellar.PRIOR_LOC[_indices]
PRIOR_SCALE = stellar.PRIOR_SCALE[_indices]
DATA_SEED = 43


def unpack(theta):
    values = [theta[0], *jnp.exp(theta[1:5]), *theta[5:8], jax.nn.sigmoid(theta[8])]
    return dict(zip(NAMES, values))


def synthetic_event():
    """Generate a new white-noise injection directly from the true lens curve."""
    model, data = stellar.synthetic_event(seed=DATA_SEED, variability=False)
    return model, data._replace(truth={name: data.truth[name] for name in NAMES})


def log_likelihood(magnification, params, flux, error):
    residual = (flux - params["f_s"] * magnification - params["f_b"]) / error
    return -0.5 * jnp.sum(residual**2 + jnp.log(2 * jnp.pi * error**2))


def negative_log_posterior(theta, model, times, flux, error):
    params = unpack(theta)
    magnification = model.magnification(times, event_parameters(params))
    return -log_likelihood(magnification, params, flux, error) + 0.5 * jnp.sum(
        ((theta - PRIOR_LOC) / PRIOR_SCALE) ** 2
    )


def fit_event(model, times, flux, error):
    """Joint MAP from data and fixed priors, without using the injected values."""

    def residual(theta):
        params = unpack(theta)
        curve = params["f_s"] * model.magnification(times, event_parameters(params)) + params["f_b"]
        return jnp.r_[(curve - flux) / error, (theta - PRIOR_LOC) / PRIOR_SCALE]

    times_np, flux_np = np.asarray(times), np.asarray(flux)
    edges = abs(times_np) > np.quantile(abs(times_np), 0.8)
    baseline = float(np.median(flux_np[edges]))
    initial = np.array(
        [0.0, np.log(0.008), np.log(14.0), np.log(0.012), 0.0, baseline - 1, 0.0, 0.0, 0.0]
    )
    evaluate, jacobian = jax.jit(residual), jax.jit(jax.jacfwd(residual))
    result = least_squares(
        lambda x: np.asarray(evaluate(x)),
        initial,
        jac=lambda x: np.asarray(jacobian(x)),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-8,
        max_nfev=400,
    )
    result.hess_inv = np.linalg.inv(result.jac.T @ result.jac)
    return result, unpack(result.x)


def experiment_metadata(model, data):
    return {
        **metadata(),
        "data_seed": DATA_SEED,
        "data_sha256": data_hash(data),
        "observations": int(data.times.size),
        "cadence_minutes": 12.0,
        "season_days": 72.0,
        "band": "W149 (reference convention)",
        "source_magnitude": 24.744,
        "source_fraction": 0.395,
        "magnitude_zeropoint": stellar.MAG_ZEROPOINT,
        "source": model.source,
        "n_fft": model.fft.n_fft,
        "fft": asdict(model.fft),
        "parallax": PARALLAX,
        "noise_model": "Independent Gaussian measurement noise; no GP or source variability",
        "error_table_sha256": hashlib.sha256(stellar.ERROR_TABLE.read_bytes()).hexdigest(),
        "injected": data.truth,
        "priors_transformed": {
            "order": list(NAMES),
            "log_parameters": list(NAMES[1:5]),
            "logit_parameters": ["a1"],
            "location": PRIOR_LOC.tolist(),
            "scale": PRIOR_SCALE.tolist(),
        },
    }
