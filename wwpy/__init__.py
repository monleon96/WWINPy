"""
WWINPy - Weight Window Python Library

A library for handling MCNP weight window input (WWINP) files.
"""

from ._config import LIBRARY_VERSION, AUTHOR

__version__ = LIBRARY_VERSION
__author__ = AUTHOR

# Import and expose the primary function
from .parser import from_file

# Only expose from_file for top-level import
__all__ = ['from_file']
