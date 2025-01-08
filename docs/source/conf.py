# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'WWPy'
copyright = '2025, Juan Antonio Monleon de la Lluvia'
author = 'Juan Antonio Monleon de la Lluvia'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',       # For automatic documentation from docstrings
    'sphinx.ext.napoleon',      # For NumPy and Google style docstrings
    'sphinx_autodoc_typehints', # For type hints support
    'myst_parser',              # For Markdown support
    'sphinx.ext.viewcode',      # Add links to source code
    'sphinx.ext.intersphinx',   # Link to other project's documentation
    'sphinx.ext.autosummary',   # Generate summary tables
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
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# HTML theme options
html_theme_options = {
    'display_version': True,
    'titles_only': False,
    'navigation_depth': 4,           
    'collapse_navigation': True,
    'sticky_navigation': True,
    'includehidden': True
}

# Document title
html_title = "WWPy"

# Theme configuration
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 3,          # Max depth of the Table of Contents
    'titles_only': False,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_with_keys': True,
    'logo_only': False,
    'display_version': True,
    'style_external_links': True,
    'titles_only': False
}

# General options
html_title = 'WWPy Documentation'
html_short_title = 'WWPy'

# Sidebar customization
html_sidebars = {
    '**': [
        'globaltoc.html',
        'localtoc.html',
        'searchbox.html',
        'relations.html',
    ]
}

# -- Path setup --------------------------------------------------------------
import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Extension configuration -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Type hints settings
always_document_param_types = True
typehints_document_rtype = True

# Make sure type hints are processed
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
add_module_names = False