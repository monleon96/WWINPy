# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

from wwpy._config import LIBRARY_VERSION, AUTHOR

# -- Project information -----------------------------------------------------
project = 'WWPy'
copyright = f"{datetime.datetime.now().year}, {AUTHOR}"
author = AUTHOR
release = LIBRARY_VERSION

# -- General configuration ---------------------------------------------------
extensions = [
    # Documentation generation
    'sphinx.ext.autodoc',       
    'sphinx.ext.napoleon',      
    'sphinx_autodoc_typehints',        # Autohints didnt work with ReadTheDocs

    # Markdown support
    'myst_parser',              

    # Additional features
    'sphinx.ext.viewcode',      
    'sphinx.ext.intersphinx',   
]

templates_path = ['_templates']
exclude_patterns = []

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Intersphinx configuration
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'navigation_depth': 3,          # Max depth of the Table of Contents
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_with_keys': True,
    'logo_only': False,
    'style_external_links': True,
    'includehidden': True,
}

html_static_path = ['_static']

# Document title
html_title = "WWPy Documentation"
html_short_title = "WWPy"

# Sidebar customization
html_sidebars = {
    '**': [
        'globaltoc.html',  # Global Table of Contents
        'searchbox.html',  # Search Box
        'relations.html',  # Next/Previous Links
    ]
}

# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# -- Autodoc Configuration ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Type hints settings
always_document_param_types = True         # Autohints didnt work with ReadTheDocs
typehints_document_rtype = True            # Autohints didnt work with ReadTheDocs

# Autodoc settings
autodoc_typehints = 'description'          # Autohints didnt work with ReadTheDocs
autodoc_member_order = 'bysource'
add_module_names = False



# -- Global variables for .rst files -------------------------------------------
rst_prolog = f"""
.. |version| replace:: {LIBRARY_VERSION}
"""