# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
from pathlib import Path
import tomllib

sys.path.insert(0, os.path.abspath("../src"))

pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
with open(pyproject_path, "rb") as f:
    project_config = tomllib.load(f)

project = 'daspal'
copyright = '2026, Florian Le Pape'
author = 'Florian Le Pape'
release = project_config["project"]["version"]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx_design",
    "myst_nb",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    ]

myst_enable_extensions = ["colon_fence"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

nb_execution_mode = "off"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

# html_sidebars = {"getting-started": [], "contribute": [], "cite": []}

html_theme = "furo"

html_static_path = ["_static/logo"]
html_favicon = "_static/logo/daspal_icon.ico"

html_theme_options = {
    "light_logo": "daspal_logo.png",
    "dark_logo": "daspal_logo_dark.png",
}

html_css_files = [
    "custom.css",
]
