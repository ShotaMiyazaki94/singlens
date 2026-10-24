singlens
========

Differentiable single-lens light curves and astrometric shifts in JAX,
with finite sources, limb darkening, and annual parallax.
Requires Python 3.12+, JAX 0.11.2+, and float64.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from singlens import AnnualParallax, Model, Parameters

   times = 8000. + jnp.linspace(-40., 40., 501)
   geometry = AnnualParallax(ra_deg=266., dec_deg=-29., t_ref=8000.)
   model = Model(source="linear", parallax=geometry)
   params = Parameters(t0=8000., u0=.05, t_e=80., rho=.1, a1=.5,
                       pi_e_n=.2, pi_e_e=-.1)
   A, shift = jax.jit(model.joint)(times, params)

Choose the source profile and geometry in ``Model``; supply fitted values
through ``Parameters``. Sources are ``point``, ``uniform``, ``linear``, and
``mu2``. For a static trajectory, omit parallax geometry and components.
See :doc:`finite-source-effects` for source profiles, solver choices, and accuracy.

See :doc:`examples/index` for photometry with a Gaussian process, flux
marginalization, and joint photometry and astrometry.

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   installation
   tutorials/quickstart
   tutorials/autodiff

.. toctree::
   :caption: Modeling
   :maxdepth: 1

   finite-source-effects
   astrometry
   examples/index

.. toctree::
   :caption: Reference
   :maxdepth: 1

   api
   numerics
   benchmarks
   theory/index
   attribution
