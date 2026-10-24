"""Shared joint-posterior initialization and a signed-impact-parameter audit."""

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares, minimize

from example.parallax_inference.code import posterior as target


def _initial(data):
    """Use the measured peak and width; never initialize from injected parameters."""
    times, flux = np.asarray(data.times), np.asarray(data.flux)
    # Most epochs are near the peak, so the median is not a baseline estimate.
    baseline = float(np.quantile(flux, 0.1))
    peak = int(np.argmax(flux))
    selected = flux > baseline + (flux[peak] - baseline) / 2
    width = float(np.ptp(times[selected])) if selected.sum() > 1 else 0.3
    theta = np.array(target.PRIOR_LOC, dtype=float)
    theta[0] = times[peak]
    theta[2] = np.log(max(width / 2, 0.01))
    return theta


class _Functions:
    """Compile once for each sign and reuse the functions across starting points."""

    def __init__(self, model, data, sign):
        def residual(theta):
            return target.residuals(theta, model, data, sign)

        def objective(theta):
            return target.negative_log_posterior(theta, model, data, sign)

        self.residual = jax.jit(residual)
        self.jacobian = jax.jit(jax.jacfwd(residual))
        self.value_and_grad = jax.jit(jax.value_and_grad(objective))
        self.hessian = jax.jit(jax.hessian(objective))


def _fit(functions, data, sign, start):
    began = perf_counter()
    dimension = len(target.NAMES)
    theta = _initial(data) if start is None else np.array(start, dtype=float, copy=True)
    if theta.shape != (dimension,) or not np.isfinite(theta).all():
        raise ValueError(f"Initial coordinates must contain {dimension} finite values")
    # Only the least-squares initializer is bounded, matching the table's radius domain.
    lower, upper = np.full(dimension, -np.inf), np.full(dimension, np.inf)
    lower[3], upper[3] = np.log(1e-6), np.log(10.0)
    theta[3] = np.clip(theta[3], lower[3] + 1e-8, upper[3] - 1e-8)
    profile = least_squares(
        lambda x: np.asarray(functions.residual(x)),
        theta,
        jac=lambda x: np.asarray(functions.jacobian(x)),
        bounds=(lower, upper),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-8,
        max_nfev=400,
    )
    if not np.isfinite(profile.fun).all() or not np.isfinite(profile.jac).all():
        raise RuntimeError("Nonfinite joint least-squares fit")
    center = profile.x
    profile_hessian = profile.jac.T @ profile.jac
    covariance = np.linalg.solve(profile_hessian, np.eye(dimension))
    scale = np.linalg.cholesky((covariance + covariance.T) / 2)

    def whitened(z):
        value, gradient = functions.value_and_grad(center + scale @ z)
        return float(value), scale.T @ np.asarray(gradient)

    result = minimize(
        whitened,
        np.zeros(dimension),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-6, "maxiter": 400},
    )
    value, gradient = whitened(result.x)
    result.optimizer_success = bool(result.success)
    result.optimizer_message = str(result.message)
    result.x = center + scale @ result.x
    result.fun = value
    result.jac = np.linalg.solve(scale.T, gradient)
    result.sign = sign
    result.whitened_gradient_max = float(np.max(np.abs(gradient)))
    result.finite_gradient = bool(np.isfinite(gradient).all())
    result.profile_nfev = int(profile.nfev)
    result.profile_success = bool(profile.success)
    result.profile_message = str(profile.message)
    result.initial_coordinates = theta
    hessian = np.asarray(functions.hessian(result.x))
    hessian = (hessian + hessian.T) / 2
    result.joint_hessian = hessian
    result.positive_hessian = False
    if np.isfinite(hessian).all():
        try:
            np.linalg.cholesky(hessian)
            covariance = np.linalg.solve(hessian, np.eye(dimension))
            result.hess_inv = (covariance + covariance.T) / 2
            np.linalg.cholesky(result.hess_inv)
            result.positive_hessian = True
        except np.linalg.LinAlgError:
            result.hess_inv = None
    else:
        result.hess_inv = None
    result.success = bool(
        np.isfinite(value)
        and result.finite_gradient
        and result.whitened_gradient_max < 1e-5
        and result.positive_hessian
    )
    result.elapsed_seconds = perf_counter() - began
    return result


def fit_event(model, data, sign=1, start=None):
    """Return a joint MAP and its exact Hessian for either fixed sign of u0.

    Both samplers use ``x`` and ``hess_inv`` from this same joint fit. All nine
    parameters, including source and blend fluxes, enter its residuals and
    Hessian. No flux parameter is profiled or marginalized out.
    ``success`` requires a finite objective, a least-squares-scaled gradient below
    1e-5, and a positive-definite joint Hessian.
    """
    if sign not in (-1, 1):
        raise ValueError("sign must be +1 or -1")
    return _fit(_Functions(model, data, sign), data, sign, start)


def _description(result, index):
    parameters = target.unpack(result.x, sign=result.sign)
    determinant = (
        float(np.linalg.slogdet(result.joint_hessian)[1]) if result.positive_hessian else None
    )
    return dict(
        start=index,
        sign=result.sign,
        success=bool(result.success),
        initial_coordinates=result.initial_coordinates.tolist(),
        coordinates=[float(value) if np.isfinite(value) else None for value in result.x],
        parameters={
            name: float(value) if np.isfinite(value) else None for name, value in parameters.items()
        },
        negative_log_posterior=float(result.fun) if np.isfinite(result.fun) else None,
        finite_gradient=result.finite_gradient,
        whitened_gradient_max=(result.whitened_gradient_max if result.finite_gradient else None),
        positive_hessian=result.positive_hessian,
        hessian_log_determinant=determinant,
        profile_nfev=result.profile_nfev,
        profile_success=result.profile_success,
        optimizer_success=result.optimizer_success,
        optimizer_message=result.optimizer_message,
        elapsed_seconds=result.elapsed_seconds,
    )


def audit_modes(model, data):
    """Return ``(report, positive_fit, negative_fit)`` from six local searches.

    Three parallax directions start each sign branch. These are reproducible
    checks for alternative solutions, not a guarantee that all modes are found.
    A failed branch returns None; every attempted optimization is retained.
    """
    began = perf_counter()
    initial = _initial(data)
    angles = np.arange(3) * (2 * np.pi / 3)
    directions = 0.2 * np.column_stack([np.cos(angles), np.sin(angles)])
    attempts, best = [], {}
    for sign in (1, -1):
        functions = _Functions(model, data, sign)
        candidates = []
        for index, direction in enumerate(directions):
            start = initial.copy()
            start[4:6] = direction
            try:
                fit = _fit(functions, data, sign, start)
                description = _description(fit, index)
                if fit.success:
                    candidates.append(fit)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                description = dict(
                    start=index,
                    sign=sign,
                    success=False,
                    initial_coordinates=start.tolist(),
                    error=str(error),
                )
            attempts.append(description)
        best[sign] = min(candidates, key=lambda fit: fit.fun) if candidates else None

    finite = [item["negative_log_posterior"] for item in attempts if item["success"]]
    for item in attempts:
        item["delta_negative_log_posterior"] = (
            item["negative_log_posterior"] - min(finite) if item["success"] else None
        )
    report = dict(
        scope="Three dispersed parallax starts per sign of u0; local MAP searches do not prove global mode completeness. Sampler timings condition on the positive-u0 branch. Both samplers share its MAP and exact joint Hessian.",
        criterion="Finite objective and gradient; max absolute least-squares-scaled gradient <1e-5; positive-definite joint Hessian",
        attempts=attempts,
        both_signs_fitted=all(fit is not None for fit in best.values()),
        elapsed_seconds_including_compile=perf_counter() - began,
    )
    if report["both_signs_fitted"]:
        positive, negative = best[1], best[-1]
        delta = float(negative.fun - positive.fun)
        log_ratio = float(
            -delta
            + 0.5
            * (
                np.linalg.slogdet(positive.joint_hessian)[1]
                - np.linalg.slogdet(negative.joint_hessian)[1]
            )
        )
        report.update(
            negative_minus_positive_nlp=delta,
            laplace_log_negative_to_positive_weight=log_ratio,
            laplace_scope="Local Gaussian volume estimate with equal prior probabilities for the two signs; not a measured mode probability or an MCMC mixing diagnostic.",
        )
    return report, best[1], best[-1]


def gradient_check(model, data, fit):
    """Compare AD with refined central differences in posterior-scaled coordinates.

    The displaced point is essential: a check only at a stationary MAP would
    mostly compare nearly zero derivatives. The scale has a unit floor, so
    each tolerance is an absolute derivative of log density per posterior-scale
    coordinate whenever the directional derivative is smaller than one.
    """
    center = jnp.asarray(fit.x)
    transform = jnp.asarray(np.linalg.cholesky(fit.hess_inv))
    sign = int(getattr(fit, "sign", 1))

    def objective(z):
        return target.negative_log_posterior(center + transform @ z, model, data, sign)

    evaluate = jax.jit(objective)
    value_and_grad = jax.jit(jax.value_and_grad(objective))
    dimension = len(target.NAMES)
    mixed = np.arange(1, dimension + 1) * (-1.0) ** np.arange(dimension)
    directions = np.vstack([np.eye(dimension), mixed / np.linalg.norm(mixed)])
    steps = 0.02 / 2 ** np.arange(5)
    positions = (
        ("map", np.zeros(dimension)),
        ("displaced", 0.5 * np.cos(np.arange(dimension))),
    )
    states = []
    for name, position in positions:
        value, gradient = value_and_grad(position)
        derivatives = directions @ np.asarray(gradient)
        central = np.array(
            [
                [
                    (
                        float(evaluate(position + step * direction))
                        - float(evaluate(position - step * direction))
                    )
                    / (2 * step)
                    for direction in directions
                ]
                for step in steps
            ]
        )
        richardson = (4 * central[1:] - central[:-1]) / 3
        scale = np.maximum(1.0, np.maximum(np.abs(derivatives), np.abs(richardson[-1])))
        difference = np.abs(derivatives - richardson[-1]) / scale
        refinement = np.abs(richardson[-1] - richardson[-2]) / scale
        finite = bool(
            np.isfinite(value)
            and np.isfinite(gradient).all()
            and np.isfinite(central).all()
            and np.isfinite(richardson).all()
            and np.isfinite(derivatives).all()
        )
        states.append(
            dict(
                name=name,
                whitened_coordinates=position.tolist(),
                negative_log_posterior=float(value) if np.isfinite(value) else None,
                ad_directional_derivatives=np.where(
                    np.isfinite(derivatives), derivatives, None
                ).tolist(),
                central_differences=np.where(np.isfinite(central), central, None).tolist(),
                richardson_derivatives=np.where(
                    np.isfinite(richardson[-1]), richardson[-1], None
                ).tolist(),
                maximum_scaled_ad_error=float(difference.max()) if finite else None,
                maximum_scaled_refinement_error=float(refinement.max()) if finite else None,
                passed=bool(finite and difference.max() < 1e-4 and refinement.max() < 1e-4),
            )
        )
    return dict(
        scope="Exact joint target in the shared MAP/Hessian affine coordinates; every coordinate direction and one mixed direction, at the MAP and a nonstationary offset. Evaluated outside sampler timers.",
        method="Central differences with five halved steps; fourth-order Richardson extrapolation from adjacent steps. Both AD agreement and final refinement must pass.",
        directions=directions.tolist(),
        steps=steps.tolist(),
        scale="max(1, abs(AD directional derivative), abs(finest Richardson derivative))",
        threshold=1e-4,
        states=states,
        passed=all(state["passed"] for state in states),
    )
