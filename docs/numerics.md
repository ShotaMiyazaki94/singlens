# Numerical accuracy

Use **float64** for finite-source calculations. The
[finite-source guide](finite-source-effects.md#validation) reports accuracy and
derivative tests for the bundled table. This page describes the optional FFTLog
backend and ephemeris limits. All error maxima are measured over sampled grids.

## FFT resolution

`solver="fft"` uses **4096 bins**. Independent checks span 31 radii from `1e-6`
to 10, including alignment, contact, and distant wings. The largest sampled
relative flux error is **0.101%**, for a uniform source near contact.
Uniform and pure linear profiles bracket `0 <= a1 <= 1`; the pure `mu2`
profile is tested separately.

More FFT bins reduce discretization error but leave the
[fixed smoothing bias](benchmark-methods.md#why-accuracy-levels-off) near
contact. See the [resolution comparison](benchmark-methods.md#fftlog-resolution).
For inference, compare flux discrepancies to measurement errors and assess
posterior sensitivity: small discrepancies can accumulate over many epochs.
Detailed [resolution checks](https://github.com/ShotaMiyazaki94/singlens/blob/main/tools/validation/README.md#fft-resolution-checks)
include value and derivative comparisons against independent integration.

### Distant observations

For `u/rho >= 20`, FFTLog uses the point-source result plus the leading
finite-source correction, preserving radius and limb-darkening dependence.
A smooth transition spans `u/rho=10..20`. Entirely distant batches skip the
FFT; mixed batches use the same distant values. Closer positions use FFTLog
for `rho >= 1e-4` and a small-source approximation below.

### Derivatives

Autodiff differentiates the numerical model. The FFT approximation boundaries
at `rho=1e-4` and `u/rho=.01` need not have continuous derivatives; the latter
uses a central expansion. Increasing FFT resolution does not test these
boundaries. Exact-contact singular slopes and table interpolation limits are
described in [finite-source validation](finite-source-effects.md#independent-accuracy-measurements).

## Astrometric shifts

Centroid shifts combine flux and image moments, so photometric accuracy does
not bound astrometric error. Use **absolute angular errors**, scaled by
`theta_E`, because shifts can pass through zero.

Independent image-moment integration tests **1,458 values** at eight radii
from `1e-6` to 10, covering alignment, approximation boundaries, distant wings,
and 41 separations over `0.99 <= u/rho <= 1.01`. Maximum FFT shift errors are:

| Profile | Error / `theta_E` |
| --- | ---: |
| Uniform | **1.241e-3** (`rho=10`, `u/rho=1.002`) |
| Pure linear (`a1=1`) | 1.581e-4 |
| Pure `mu2` | 1.956e-5 |

At `rho=.1`, `u/rho=.5`, the uniform-source radius derivative differs from
independent finite differences by **2%** with 4096 bins. In this case, 8192
bins gives radius and position derivatives within `1e-7` relative. These
measurements illustrate why value and derivative accuracy need separate checks.

The distant expansion retains the astrometric `1/u` tail. Alignment, contact,
and rotational identities are given in [theory](theory/index.md#astrometric-centroid-shift).
Reproduce the checks with `python -m tools.validation.astrometry_accuracy`;
[validation details](https://github.com/ShotaMiyazaki94/singlens/blob/main/tools/validation/README.md)
record the reference methods and regression thresholds.

## Annual parallax

The daily Earth ephemeris covers **2000–2050**; its sampling limits time
derivatives. Out-of-range queries, including after HJD conversion, return NaN.
Keep the geocentric reference epoch fixed when fitting `t0`. See
[API time conventions](api.md#annual-parallax).
