# Configuration file for the Sphinx documentation builder.
import os
import sys
from datetime import date

sys.path.insert(0, '..')
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- General configuration -----------------------------------------------

project = 'GPFA Documentation'
authors = 'Brooks M. Musangu and Jan Drugowitsch'
copyright = "2021-{this_year}, {authors}".format(
    this_year=date.today().year, authors=authors)
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.imgmath',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    # 'nbsphinx',
    # 'sphinx_tabs.tabs',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# The reST default role (used for this markup: `text`) to use for all 
# documents.
# default_role = None

# path to bibliography
bibtex_bibfiles = ['./bib/gpfa.bib']

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
# add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
# show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

katex_prerender = True

# Only execute Jupyter notebooks that have no evaluated cells
nbsphinx_execute = 'auto'
# Kernel to use for execution
nbsphinx_kernel_name = 'python3'
# Cancel compile on errors in notebooks
nbsphinx_allow_errors = False

# Required to automatically create a summary page for each function listed in
# the autosummary fields of each module.
autosummary_generate = True

# Set to False to not overwrite the custom _toctree/*.rst
autosummary_generate_overwrite = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

# html_theme_options = {
#     'display_version': True,
#     'prev_next_buttons_location': 'bottom',
#     'style_external_links': False,
#     'vcs_pageview_mode': '',
#     # Toc options
#     'collapse_navigation': False,
#     'sticky_navigation': True,
#     'includehidden': True,
# }

html_static_path = ['_static']

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
# html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
# html_use_opensearch = ''

# This is the file name suffix for HTML files (e.g. ".xhtml").
# html_file_suffix = None

# Output file base name for HTML help builder.
htmlhelp_basename = 'gpfadoc'

# Suppresses  wrong numpy doc warnings
# see here https://github.com/phn/pytpm/issues/3#issuecomment-12133978
numpydoc_show_class_members = False

# # Use more reliable mathjax source
mathjax_path = 'https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML'
imgmath_image_format = "svg"

# # -- Options for LaTeX output ------------------------------------------------
latex_elements = {

}
    
latex_documents = [
    ('index', 'gpfa.tex', 'GPFA Documentation',
     authors, 'manual'),
]
# latex_elements = {
#     # The paper size ('letterpaper' or 'a4paper').
#     # 'papersize': 'letterpaper',
#     # The font size ('10pt', '11pt' or '12pt').
#     # 'pointsize': '10pt',
#     # Additional stuff for the LaTeX preamble.
#     "preamble": r"""
#         \usepackage{amsmath}\usepackage{amsfonts}\usepackage{bm}
#         \usepackage{morefloats}\usepackage{enumitem} \setlistdepth{10}
#         \let\oldhref\href
#         \renewcommand{\href}[2]{\oldhref{#1}{\hbox{#2}}}
#         """
# }
# # For maths, use mathjax by default and svg if NO_MATHJAX env variable is set
# # (useful for viewing the doc offline)
# if os.environ.get("NO_MATHJAX"):
#     extensions.append("sphinx.ext.imgmath")
#     imgmath_image_format = "svg"
#     mathjax_path = ""
# else:
#     extensions.append("sphinx.ext.mathjax")
#     mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"