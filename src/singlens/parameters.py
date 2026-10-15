"""Named, differentiable event parameters, independent of solver configuration."""

from dataclasses import dataclass, fields, replace

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, kw_only=True)
class Parameters:
    """Physical parameters of one event, registered as a JAX pytree.

    ``t0`` and ``t_e`` use the same time units as the observations. ``u0`` and
    ``rho`` are in Einstein-radius units. ``a1`` is the linear limb-darkening
    coefficient, and ``pi_e_n``/``pi_e_e`` are the north/east parallax components.

    Omit parameters not used by the model. Numeric inputs become floating-point
    arrays so even integer literals work with ``jax.grad``. Batched parameters
    are allowed for ``jax.vmap``; each model evaluation takes scalar parameters.
    Use ``replace`` to vary parameters without mutating an existing event.
    """

    t0: ArrayLike
    u0: ArrayLike
    t_e: ArrayLike
    rho: ArrayLike | None = None
    a1: ArrayLike | None = None
    pi_e_n: ArrayLike | None = None
    pi_e_e: ArrayLike | None = None

    def __post_init__(self):
        if self.t0 is None or self.u0 is None or self.t_e is None:
            raise ValueError("t0, u0, and t_e are required.")
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None:
                array = jnp.asarray(value)
                if jnp.iscomplexobj(array):
                    raise ValueError(f"{field.name} must be real.")
                object.__setattr__(self, field.name, array.astype(jnp.result_type(array, float)))

    def replace(self, **changes):
        """Return a new parameter set, e.g. ``params.replace(rho=0.02)``."""
        return replace(self, **changes)

    def tree_flatten(self):
        return tuple(getattr(self, field.name) for field in fields(self)), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # JAX may reconstruct trees with sentinels or Jacobian arrays; do not
        # rerun input conversion on these internal representations.
        result = object.__new__(cls)
        for field, value in zip(fields(cls), children):
            object.__setattr__(result, field.name, value)
        return result
