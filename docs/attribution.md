# Citation and licenses

singlens is based on [microJAX](https://github.com/ShotaMiyazaki94/microjax).
For scientific use, cite Miyazaki & Kawahara (2025),
[ApJ 994, 144](https://doi.org/10.3847/1538-4357/ae1005), and the numerical
method used. Report the singlens version, solver, JAX version, device, and precision.

The bundled table is generated using the disk-mixture identity of
[Wang (2026), arXiv:2607.12483](https://arxiv.org/abs/2607.12483).
See [finite-source effects](finite-source-effects.md) for its accuracy and
provenance. The FFT backend follows
[Sugiyama (2022), arXiv:2203.06637](https://arxiv.org/abs/2203.06637).

The Gould–Yoo kernel follows Gould (1994) and Yoo et al. (2004),
with its normalized B0/B1 formulation also given in
[the OGLE 2004-BLG-482 limb-darkening analysis](https://doi.org/10.1051/0004-6361/200912007).
The finite-source astrometric moment construction follows
[Mao & Witt (1998)](https://arxiv.org/abs/astro-ph/9804045).

The original Roman recovery setup and retained magnitude-error table come from
[test_microjax/Roman_recovery](https://github.com/ShotaMiyazaki94/test_microjax/tree/979fa9b33b69c4ba89eb7ad70ebae3d23da349cc/Roman_recovery).
Physical examples use CK04 stellar atmospheres and Roman F146 effective areas;
their {download}`input provenance <../example/inference/data/active_reference_provenance.json>`
records sources, versions, and checksums.
The CK04-derived CSV is fetched locally with
`python tools/fetch_example_data.py --bootstrap`; it is excluded from Git and
source distributions because upstream redistribution permission is unconfirmed.
The helper verifies both original STScI downloads and the derived table against
the recorded hashes.
GP inference uses [tinygp](https://tinygp.readthedocs.io/en/stable/).
External benchmarks use [VBMicrolensing](https://github.com/valboz/VBMicrolensing),
[MulensModel](https://github.com/rpoleski/MulensModel), and
[emcee](https://emcee.readthedocs.io/en/stable/).

Original singlens code is MIT licensed. Derived components carry MIT,
Apache-2.0, and BSD-3-Clause licenses; see `LICENSES/` and `NOTICE` for the
required notices. Source provenance is available in [upstream.json](upstream.json).
