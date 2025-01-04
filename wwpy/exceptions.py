# wwpy/exceptions.py

class WWINPFormatError(Exception):
    """Raised when the WWINP file is incorrectly formatted."""
    pass

class WWINPParsingError(Exception):
    """Raised for general parsing errors in the WWINP file."""
    pass
