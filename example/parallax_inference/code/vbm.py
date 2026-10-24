"""VBMicrolensing magnifications for the common parallax/flux posterior."""

from importlib import import_module
from time import perf_counter

import jax
import numpy as np
from scipy.special import expit

from example._cpu import cpu_seconds
from example.parallax_inference.code import experiment as exp
from example.parallax_inference.code import posterior as target


class VBMPosterior:
    """Exchange only the finite-source engine; retain geometry and likelihood."""

    def __init__(self, data, *, tolerance=1e-3, minannuli=2):
        started = perf_counter()
        cpu_started = cpu_seconds()
        import VBMicrolensing

        adapter = import_module("example.compare-single-vbml.code.vbm_array")
        self.engine = adapter.VBMArray(tolerance=tolerance, minannuli=minannuli)
        self.stock = VBMicrolensing.VBMicrolensing()
        self.stock.Tol = self.stock.RelTol = tolerance
        self.stock.minannuli = minannuli
        self.data = data
        self.geometry = exp.Geometry(*(np.asarray(value) for value in data.geometry))
        self.log_prob = jax.jit(
            jax.vmap(
                lambda theta, values: (
                    target.log_likelihood(theta, values, data) + target.log_prior(theta)
                )
            )
        )
        self.setup_seconds = perf_counter() - started
        self.setup_cpu_seconds = cpu_seconds() - cpu_started
        self.pool = None
        self.metadata = dict(
            solver="VBMicrolensing",
            source="linear",
            tolerance=tolerance,
            relative_tolerance=tolerance,
            minannuli=minannuli,
            interface=self.engine.metadata,
            trajectory="Common Earth-geocentric annual-parallax exposure nodes",
        )

    def start_workers(self, workers):
        if workers > 1:
            from example.parallax_inference.code.vbm_pool import VBMProcessPool

            self.pool = VBMProcessPool(
                workers, self.metadata["tolerance"], self.metadata["minannuli"]
            )
            self.metadata["parallel_execution"] = self.pool.metadata
        else:
            self.metadata["parallel_execution"] = dict(workers=1, start_method="in-process")

    @property
    def active_worker_cpu_seconds(self):
        return self.pool.active_cpu_seconds if self.pool is not None else 0.0

    def stop_workers(self):
        if self.pool is not None:
            self.pool.close()
            self.pool = None

    def positions(self, theta, geometry=None):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        if theta.shape[1] != len(target.NAMES):
            raise ValueError("Expected nine transformed parameters")
        geometry = self.geometry if geometry is None else geometry
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            rho = np.exp(theta[:, 3])
            t_e = np.exp(theta[:, 2]) / rho
            u0 = rho * np.exp(theta[:, 1])
            tau = (np.asarray(geometry.times) - theta[:, 0, None, None]) / t_e[:, None, None]
            tau += theta[:, 4, None, None] * np.asarray(geometry.tau_n)
            tau += theta[:, 5, None, None] * np.asarray(geometry.tau_e)
            beta = u0[:, None, None] + theta[:, 4, None, None] * np.asarray(geometry.beta_n)
            beta += theta[:, 5, None, None] * np.asarray(geometry.beta_e)
        return rho, expit(theta[:, 6]), np.hypot(tau, beta)

    def magnifications(self, theta, *, geometry=None, stock=False):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        geometry = self.geometry if geometry is None else geometry
        rho, a1, positions = self.positions(theta, geometry)
        u = positions.reshape(len(theta), -1)
        if stock:
            values = []
            for coordinates, coefficient, separation in zip(theta, a1, u):
                self.stock.a1 = float(coefficient)
                # Supplying separations as times evaluates the same ESPLMag2
                # core with u0=0, tE=1, t0=0; parallax is already in separation.
                values.append(
                    self.stock.ESPLLightCurve(
                        [-np.inf, 0.0, 0.0, float(coordinates[3])], separation.tolist()
                    )[0]
                )
            values = np.asarray(values)
        else:
            engine = self.engine if self.pool is None else self.pool
            values = engine.magnifications(rho, a1, u)
        return values.reshape(positions.shape) @ np.asarray(geometry.weights)

    def __call__(self, theta):
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        valid = (
            np.isfinite(theta).all(axis=1)
            & (theta[:, 3] >= np.log(1e-6))
            & (theta[:, 3] <= np.log(10))
        )
        result = np.full(len(theta), -np.inf)
        candidates = np.flatnonzero(valid)
        if len(candidates):
            rho, _, positions = self.positions(theta[candidates])
            domain = np.isfinite(positions).all(axis=(1, 2)) & np.all(
                positions <= 1e8 * rho[:, None, None], axis=(1, 2)
            )
            selected = candidates[domain]
            if len(selected):
                result[selected] = self.log_prob(
                    theta[selected], self.magnifications(theta[selected])
                )
        return np.where(np.isfinite(result), result, -np.inf)


def validate_vbm(posterior, data, states):
    """Measure native accuracy separately from interface/exposure correctness."""
    independent, exposure, interface = [], [], []
    refined = VBMPosterior(data, tolerance=1e-6, minannuli=4)
    for index, state in enumerate(states):
        theta = target.pack(state)[None]
        params = {name: state[name] for name in exp.LENS_NAMES}
        selected = exp._validation_epochs(data, params)
        times = np.asarray(data.times)[selected]
        durations = np.asarray(data.exposure_seconds)[selected]
        geometry = [exp.prepare_geometry(times, durations, order=order) for order in (3, 9, 15)]
        coarse, middle, fine = [posterior.magnifications(theta, geometry=g)[0] for g in geometry]
        stock = posterior.magnifications(theta, geometry=geometry[0], stock=True)[0]
        interface.append(float(np.max(np.abs(coarse / stock - 1))))
        scale = abs(float(state["f_s"])) / np.asarray(data.error)[selected]
        exposure.append(
            dict(
                state=index,
                epochs=selected.tolist(),
                three_minus_fifteen_sigma=float(np.max(abs(coarse - fine) * scale)),
                nine_minus_fifteen_sigma=float(np.max(abs(middle - fine) * scale)),
            )
        )
        if index < 2:
            reference = exp.reference_magnification(
                times, durations, params, order=15, spatial_order=64
            )
            tighter = refined.magnifications(theta, geometry=geometry[0])[0]
            independent.append(
                dict(
                    state=index,
                    epochs=selected.tolist(),
                    max_abs_fit_minus_independent_sigma=float(
                        np.max(abs(coarse - reference) * scale)
                    ),
                    max_relative_magnification_difference=float(
                        np.max(abs(coarse / reference - 1))
                    ),
                    max_abs_tighter_minus_independent_sigma=float(
                        np.max(abs(tighter - reference) * scale)
                    ),
                )
            )
    theta = np.stack([target.pack(state) for state in states])
    native, tight = posterior.magnifications(theta), refined.magnifications(theta)
    delta = np.asarray(posterior.log_prob(theta, native) - posterior.log_prob(theta, tight))
    finite = bool(
        np.isfinite(native).all() and np.isfinite(tight).all() and np.isfinite(delta).all()
    )
    maximum_error = max(item["max_abs_fit_minus_independent_sigma"] for item in independent)
    maximum_exposure = max(item["three_minus_fifteen_sigma"] for item in exposure)
    return dict(
        passed=finite and max(interface) < 1e-12 and maximum_exposure < 0.01,
        scope="Native VBM settings; numerical approximation errors are measured, not forced to match Table accuracy. Interface and finite-exposure correctness are checked separately.",
        max_abs_fit_minus_independent_sigma=maximum_error,
        max_abs_three_minus_fifteen_sigma=maximum_exposure,
        approximation_within_0_01_sigma=maximum_error < 0.01,
        max_relative_array_minus_stock=max(interface),
        max_abs_native_minus_tight_log_posterior=float(np.max(abs(delta))),
        native_minus_tight_log_posterior_range=[float(delta.min()), float(delta.max())],
        independent=independent,
        exposure_refinement=exposure,
        tighter_settings=dict(tolerance=1e-6, relative_tolerance=1e-6, minannuli=4),
    )
