"""Differentiable single-lens models. Enable JAX x64 before constructing a Model."""

from importlib.metadata import PackageNotFoundError, version

from .model import Model
from .parameters import Parameters
from .photometry import FluxFit
from .trajectory import AnnualParallax

try:
    __version__ = version("singlens")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "Model",
    "Parameters",
    "AnnualParallax",
    "FluxFit",
]
