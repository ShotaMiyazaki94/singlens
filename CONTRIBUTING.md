# Development

Clone the repository and create an environment with Python 3.12 or later:

```bash
git clone https://github.com/ShotaMiyazaki94/singlens.git
cd singlens
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the development dependencies and run the checks from the repository root:

```bash
python -m pip install -e '.[dev,docs,validation]'
ruff check src tests example tools docs/conf.py
ruff format --check src tests example tools docs/conf.py
python -m pytest
python -m sphinx -E -b html -W --keep-going docs docs/_build/html
python -m build
python -m twine check --strict dist/*
```

CI runs lint, the test suite, and distribution checks once on Python 3.12
with the minimum supported JAX version. Optional example and reference-library
dependencies are installed so those tests also run.

Build documentation locally before submitting changes to docs or examples.
Show the scientific calculation in tutorials: define the probability model,
likelihood, sampler, and predictions directly. Keep file loading and global
plot styling out of the rendered code. Saved long-run samples may supply figures;
state which run they represent. The inference tutorial includes the actual
example model and sampler source with `literalinclude`, so the displayed
definitions and executed implementation cannot drift apart.
Its explicit `RUN_SAMPLING` switch selects a new run or saved draws; never
infer that choice from which variables happen to exist in the kernel.
Sphinx executes three notebooks and treats warnings/errors as failures.
`-E` also reruns notebooks whose imported helpers changed. Open
`docs/_build/html/index.html`. Numerical diagnostics live in `tools/validation`;
regenerate example results when changing the solver. Benchmark separately
from tests or other CPU-intensive work.

The wheel contains the Earth ephemeris and license notices. Source archives
include examples and tests, but exclude historical `archive/`, generated docs,
and development outputs. Test a built wheel in a fresh environment before release.
Version metadata lives in `pyproject.toml`.

## Repository contents

Keep source, tests, documentation, examples, validation scripts, and license
notices under version control. The figures, data, reports, and posterior
samples used by the public examples are also versioned so the documentation
can build without rerunning long inference jobs.

`.gitignore` excludes local archives and experiments, virtual environments,
caches, builds, private settings, numerical diagnostic outputs, and temporary
checkpoints. `MANIFEST.in` applies the same
exclusions to source distributions. Keep ad hoc work in the ignored `local/`
directory and original snapshots in the ignored `archive/` directory.

## Implementation layout

The package root holds the event API (`lightcurve.py`), prepared `Model`, and
`Parameters`. Physical and numerical work is grouped by responsibility:

- `lensing/`: point/finite-source magnification and `FFTConfig`.
- `trajectory/`: `AnnualParallax`, ephemeris loading/projection, and Kepler geometry.
- `photometry/`: weighted flux fitting and Gaussian marginal likelihoods.
- `numerics/`: internal FFTLog, interpolation, and Bessel kernels.
- `data/`: bundled Earth ephemeris.

Keep numerical kernels independent of lensing and event APIs. Public entry
points are exported from `singlens`.

## Numerical maintenance

Public docs describe behavior, usage, measured performance, and accuracy limits.
Document current algorithms and validation procedures here or in
`tools/validation`; keep dated working notes and superseded results locally.

Finite-source interpolation uses a natural C2 cubic spline at interior grid
knots. The far-field expansion is joined over `u/rho=10..20` with matching
first and second derivatives. Test that all-far batches skip the FFT under
both evaluation and autodiff, and give the same distant values as mixed
batches. Other boundaries remain at `rho=1e-4`, the central expansion at
`u/rho=.01`, and the FFT grid's constant extension at `u=100` before blending.
Keep independent value, slope, and curvature checks when changing these paths.

GP recovery uses a NumPyro model and one fixed-length `MCMC.run`, initialized
from the joint MAP and its covariance. The speed comparison keeps its ESS
stopping loop in `example/compare-single-vbml/code/sampling.py`; its affine
transformation has a constant Jacobian. Diagnostics and IO are excluded from
both benchmark samplers' timings. The speed comparison generates independent
noise directly; GP injection–recovery remains a separate example.

The [VBM inference adapter](docs/benchmark-methods.md#vbm-inference-interface)
must keep calls serial and the installed numerical core unchanged, with no
silent interface fallback. Include its build/load cost once in VBM's total
and check agreement with the stock API outside sampling timers. Record the
interface and source provenance in the report. Regenerate all three inference
runs together when publishing new speed ratios, without concurrent intensive work.

The independent accuracy reference cancels the point-lens singularity with
the lens-centered area element. Its convergence and center checks are in
`example/compare-single-vbml/code/accuracy.py`. The default-domain check is
`python -m tools.validation.accuracy_domain`; it caches independent references
and checks 31 radii, fine source-contact sampling, approximation boundaries,
and source-centered wing integrals.

VBM uses `minannuli=2` for linear limb darkening to avoid stopping when disk
and point-source magnifications nearly coincide. Runtime comparisons retain
each reference package's native approximations. A nominal VBM tolerance is
not an accuracy bound; report measured integration errors and numerical
posterior differences separately from timing.

### FFT domain and regularization

[Sugiyama (2022), Sections III–IV](https://arxiv.org/html/2203.06637v1#S3)
demonstrates `u_i=10**(-6+9*i/N)` with `N=1024` and a Gaussian high-frequency
filter with cutoff `50*k_rho`, where `k_rho=2*pi/rho`. It does not establish
an optimal transform domain.

The default grid is inclusive `logspace(-8,4,4096)`. The full calculation and
small-source limit share the compensated Gaussian `2*G-G**2`, with
`G=exp[-(k*rho/(100*pi))**2]`. Its low-frequency expansion starts at fourth
order, reducing smoothing bias while suppressing undersampled high-frequency
modes. The first point-lens transform also restores the omitted interval
`0<u<u_min`, using `A_p=1/u+O(u)` and the convergent Bessel series at
`k*u_min<=1/2`. These corrections improve both values and derivatives without
adding transforms to each evaluation.

Above 8192 bins, the inverse transform uses padding proportional to resolution
to prevent wrap-around near the central expansion.

The 4096-bin default accepts a 0.11% uniform-source validation threshold near
limb contact; pure linear and quadratic basis checks retain 0.1% thresholds.
Public accuracy figures report the measured errors rather than these check limits.

For small sources, `A_ext=A_ext0(u/rho)/rho+O(rho)` has no additive constant.
The far-field branch begins at `u/rho=20` for every source radius. It retains
the second-order correction from the brightness profile's radial moment;
the uncorrected point-source error is about 0.031% there in the high-
magnification uniform-disk limit.

Near alignment, the quadratic central expansion avoids FFT endpoint errors.
A radial profile can be written as a weighted mixture of uniform disks;
their exact central curvature, `A_p'(rho)/rho`, gives the second-order term.
The existing `r=sin(theta)` quadrature removes the limb endpoint singularity
for the nonuniform basis profiles. Tests check both near-center values and
slopes against independent integration, and uniform-source derivatives against
the elliptic small-source limit.

FFTLog uses JAX complex `loggamma`; the Bessel approximations have broad-argument
tests because `bessel_jn` is not a drop-in replacement across that range.

## Publication

`.readthedocs.yaml` supplies the documentation build configuration. On Read the
Docs Community, import the public `ShotaMiyazaki94/singlens` repository, select
`main` as the default branch, and use the project slug `singlens` if available.
The documentation URL is `https://singlens.readthedocs.io/en/latest/`; update the
README and package metadata if a different slug is required. Read the Docs
builds documentation when changes are pushed.

PyPI publication uses [Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
through `.github/workflows/publish.yml`. Set up the accounts once:

1. In the GitHub repository settings, create an environment named `pypi`.
2. In PyPI account settings, add a pending GitHub publisher with project name
   `singlens`, owner `ShotaMiyazaki94`, repository `singlens`, workflow filename
   `publish.yml`, and environment `pypi`.

After the checks pass, publish a GitHub Release with a tag matching the version
in `pyproject.toml`, for example `v0.1.0` for version `0.1.0`. The publication
workflow verifies that match, builds and checks the wheel and source archive,
and uploads both to PyPI. No API token is stored in GitHub. The first successful
upload creates the PyPI project; a pending publisher does not reserve its name.
Use a new version for subsequent releases. Regular pushes only run CI.
