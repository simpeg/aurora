# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import aurora
from sphinx_gallery.sorting import FileNameSortKey

# -- Project information -----------------------------------------------------

project = "aurora"
copyright = "2021, Karl Kappler, Jared Peacock, Andy Frassetto, Tim Ronan, Lindsey Heagy, Douglas Oldenburg"
author = "2021, Karl Kappler, Jared Peacock, Tim Ronan, Andy Frassetto, Lindsey Heagy, Douglas Oldenburg"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "matplotlib.sphinxext.plot_directive",
    "numpydoc",
    "nbsphinx",
    "sphinx_gallery.gen_gallery",
]

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
autosummary_generate = True

numpydoc_class_members_toctree = False

# API doc options
apidoc_module_dir = "../aurora"
apidoc_output_dir = "api/generated"
apidoc_toc_file = False
apidoc_excluded_paths = []
apidoc_separate_modules = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
try:
    import sphinx_rtd_theme

    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
    pass
except Exception:
    html_theme = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/", None),
}

# Sphinx Gallery
sphinx_gallery_conf = {
    # path to your examples scripts
    "examples_dirs": [
        "../tutorials",
    ],
    "gallery_dirs": [
        "examples",
    ],
    "within_subsection_order": FileNameSortKey,
    "filename_pattern": "\.py",
    "backreferences_dir": "api/generated/backreferences",
    "doc_module": "aurora",
    # 'reference_url': {'discretize': None},
}
