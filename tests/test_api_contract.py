"""The public root exposes explicit event models and their configuration."""

import importlib.util

import pytest

import singlens as sl
import singlens.lensing as lensing
from singlens.lensing import FFTConfig


def test_public_root_exports():
    expected = {
        "__version__",
        "Model",
        "Parameters",
        "AnnualParallax",
        "FluxFit",
    }
    assert set(sl.__all__) == expected
    assert all(hasattr(sl, name) for name in expected)


def test_model_requires_an_explicit_keyword_source():
    with pytest.raises(TypeError, match="source"):
        sl.Model()
    with pytest.raises(TypeError):
        sl.Model("point")
    with pytest.raises(ValueError, match="source must"):
        sl.Model(source=None)


def test_model_solver_configuration_has_one_keyword():
    with pytest.raises(TypeError, match="fft"):
        sl.Model(source="linear", fft=FFTConfig())


def test_generation_quadrature_is_not_a_public_solver():
    assert "ConcentricDisk" not in lensing.__all__
    with pytest.raises(AttributeError):
        getattr(lensing, "ConcentricDisk")


@pytest.mark.parametrize(
    "name",
    [
        "magnification",
        "astrometric_shift",
        "fit_flux",
        "marginalize_flux",
        "pspl",
        "image_positions",
        "UniformDisk",
        "LinearLimbDarkened",
        "QuadraticProfile",
        "UniformDiskQuadrature",
        "GouldYoo",
        "ConcentricDisk",
        "FFTConfig",
        "BasisTable",
    ],
)
def test_removed_root_names_are_unavailable(name):
    with pytest.raises(AttributeError):
        getattr(sl, name)


def test_function_wrapper_module_is_unavailable():
    assert importlib.util.find_spec("singlens.lightcurve") is None
