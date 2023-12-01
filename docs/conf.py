# Configuration file for the Sphinx documentation builder.
import sys
import os
from datetime import date

sys.path.insert(0, os.path.abspath('../'))
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Path to the source directory (where your .rst files are)
sourcedir = os.path.abspath(os.path.dirname(__file__))

# Path to the build directory (where your HTML output will go)
builddir = os.path.abspath(os.path.join(os.path.dirname(__file__), '_build'))

sys.path.append(os.path.abspath('sphinxext'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- General configuration -----------------------------------------------

project = 'GPFA Documentation'
authors = 'Brooks M. Musangu and Jan Drugowitsch'
copyright = "2021-{this_year}, {authors}".format(
    this_year=date.today().year, authors=authors)
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "numpydoc",
]

# Required to automatically create a summary page for each function listed in
# the autosummary fields of each module.
autosummary_generate = True

templates_path = ['_templates']
# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build", "templates", "includes", "themes"]

# The suffix of source filenames.
source_suffix = '.rst'

# The root toctree document.
root_doc = 'index'

# Intersphinx allows you to link to the API reference of another project
# (as long as the documentation for that project was also built with Sphinx).
intersphinx_mapping = {
    "python": ("https://docs.python.org/{.major}".format(sys.version_info), None),
    "numpy": ('http://docs.scipy.org/doc/numpy/', None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_static_path = ['_static']

html_theme_options = {
    'analytics_id': 'UA-12345678-9',
}

autodoc_member_order = ['groupwise']

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# If false, no index is generated.
html_use_index = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# Create table of contents entries for domain objects (e.g. functions,
# classes, attributes, etc.). Default is True.
toc_object_entries = False
