"""Custom exceptions for WWINP file handling.

Defines specific exception types for format and parsing errors.
"""

# wwinpy/exceptions.py

class WWINPFormatError(Exception):
    """Exception raised for WWINP file format errors.

    :raises: When the WWINP file structure is invalid
    """
    pass

class WWINPParsingError(Exception):
    """Exception raised for WWINP parsing errors.

    :raises: When the WWINP file content cannot be parsed
    """
    pass
