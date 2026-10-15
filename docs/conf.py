"""Sphinx configuration; notebook errors and build warnings fail CI."""

import os
from pathlib import Path

import singlens

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
# Notebook kernels change CWD; preserve a caller's relative cache path.
os.environ["MPLCONFIGDIR"] = str(Path(os.environ["MPLCONFIGDIR"]).expanduser().resolve())
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("IPYTHONDIR", str(ROOT / "docs/_build/.ipython"))

# Prepare the shared font cache before notebook stderr is treated as a warning.
# This also makes clean CI builds independent of a pre-existing user cache.
import matplotlib.font_manager  # noqa: E402, F401

project = "singlens"
author = "Shota Miyazaki"
copyright = "2026, Shota Miyazaki"
release = singlens.__version__
version = release
language = "en"
root_doc = "index"

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_nb",
    "sphinx.ext.autodoc",
]
source_suffix = {".rst": "restructuredtext", ".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = ["_build", "_autoapi", ".jupyter_cache", "**/.ipynb_checkpoints"]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
nb_execution_mode = "force"
# Prefer this environment over a stale user-level kernel with the same name.
os.environ.setdefault("JUPYTER_PREFER_ENV_PATH", "1")
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_execution_allow_errors = False
nb_execution_show_tb = True
nb_execution_in_temp = False
nb_output_stderr = "warn"

autodoc_typehints = "none"
napoleon_numpy_docstring = True
napoleon_google_docstring = True

html_theme = "sphinx_book_theme"
html_title = "singlens"
html_logo = "../logo_transparent.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_extra_path = ["upstream.json"]
html_theme_options = {"show_navbar_depth": 2, "show_toc_level": 2, "use_download_button": True}
html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "")
if os.environ.get("READTHEDOCS") == "True":
    html_context = {"READTHEDOCS": True}
