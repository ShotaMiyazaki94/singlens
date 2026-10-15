singlens
========

Single-lens microlensing in JAX. Pass observation times and microlensing
parameters to ``magnification`` to compute static or annual-parallax light
curves, then differentiate or fit the same function.

Point sources, uniform disks, and linear limb darkening are supported.
Use Python 3.12+, JAX 0.11.2+, and float64.

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)
   import jax.numpy as jnp
   from singlens import magnification, AnnualParallax

   times = 8000. + jnp.linspace(-40., 40., 501)
   geometry = AnnualParallax(ra_deg=266., dec_deg=-29., t_ref=8000.)
   A = magnification(times, t0=8000., u0=.05, t_e=80., rho=.1, a1=.5,
                     parallax=geometry, pi_e_n=.2, pi_e_e=-.1)

Omit ``rho`` for a point source or ``a1`` for a uniform disk. For a static
trajectory, omit the parallax inputs.

.. toctree::
   :maxdepth: 1

   installation
   tutorials/quickstart
   tutorials/autodiff
   tutorials/inference
   api
   benchmarks
   numerics
   theory/index
   attribution
