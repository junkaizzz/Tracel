"""Sphinx configuration for the Tracel documentation (Read the Docs theme)."""
import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "tracel"
author = "Junkai Zhang, Hao Chen"
copyright = "2026, Junkai Zhang and Hao Chen"
release = version = "0.1.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
]
autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
# Heavy numerical dependencies are mocked so the documentation builds without installing them.
autodoc_mock_imports = ["numpy", "pandas", "scipy", "anndata", "scanpy", "pyarrow", "jax", "ott", "moscot"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_theme_options = {"logo_only": False, "navigation_depth": 3}
html_title = "tracel documentation"
