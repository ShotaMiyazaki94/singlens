# Event derivatives

```bash
python example/jacobian-single/code/autodiff.py
```

With the `examples` extra installed, run from the repository root.
`outputs/jacobian.png` and `.pdf` show annual-parallax light curves and
Jacobian components for `t0`, `t_e`, `u0`, `rho`, `pi_e_n`, `pi_e_e`, and
the linear limb coefficient `a1`. Uniform and linear sources share the same
geometry; only the linear source has an `a1` derivative.

The event uses HJD − 2450000: `t0=8000`, `u0=.05`, `t_e=80` days,
`rho=.1`, `pi_e_n=.2`, `pi_e_e=-.1`, and `a1=.5`.
RA 266°, Dec −29°, and `t_ref=8000` remain fixed during differentiation.
Use `--output-dir PATH` to change the destination.
