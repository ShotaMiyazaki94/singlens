# Citation and licenses

singlens is based on [microJAX](https://github.com/ShotaMiyazaki94/microjax).
For scientific use, cite Miyazaki & Kawahara (2025),
[ApJ 994, 144](https://doi.org/10.3847/1538-4357/ae1005), and the finite-source
FFT method by Sugiyama (2022), [arXiv:2203.06637](https://arxiv.org/abs/2203.06637).
Report the singlens version, FFT configuration, JAX version, device, and precision.

The Roman-like example's cadence, source/blend photometry, and magnitude-error
table come from
[test_microjax/Roman_recovery](https://github.com/ShotaMiyazaki94/test_microjax/tree/979fa9b33b69c4ba89eb7ad70ebae3d23da349cc/Roman_recovery).
GP inference uses [tinygp](https://tinygp.readthedocs.io/en/stable/).
External benchmarks use [VBMicrolensing](https://github.com/valboz/VBMicrolensing),
[MulensModel](https://github.com/rpoleski/MulensModel), and
[emcee](https://emcee.readthedocs.io/en/stable/).

Original singlens code is MIT licensed. Derived components carry MIT,
Apache-2.0, and BSD-3-Clause licenses; see `LICENSES/` and `NOTICE` for the
required notices. Source provenance is available in [upstream.json](upstream.json).
