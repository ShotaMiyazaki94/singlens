"""Differentiable single-lens microlensing. Enable JAX x64 in your application."""

from importlib.metadata import PackageNotFoundError, version

from .lensing import FFTConfig, LinearLimbDarkened, QuadraticProfile, UniformDisk
from .lensing.point import image_positions, pspl
from .lightcurve import magnification
from .model import Model
from .parameters import Parameters
from .photometry import FluxFit, fit_flux
from .trajectory import AnnualParallax

try:
    __version__ = version("singlens")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "__version__",
    "magnification",
    "Model",
    "Parameters",
    "FFTConfig",
    "AnnualParallax",
    "FluxFit",
    "fit_flux",
    "pspl",
    "image_positions",
    "UniformDisk",
    "LinearLimbDarkened",
    "QuadraticProfile",
]
