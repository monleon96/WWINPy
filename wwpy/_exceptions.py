"""
Custom exceptions for the WWPy package, providing specific error types for WWINP file handling and parsing.
These exceptions help in better error handling and debugging of weight window input files.
"""

# wwpy/exceptions.py

class WWINPFormatError(Exception):
    """Raised when the WWINP file is incorrectly formatted."""
    pass

class WWINPParsingError(Exception):
    """Raised for general parsing errors in the WWINP file."""
    pass
