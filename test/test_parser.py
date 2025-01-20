# test_parser.py

import pytest
from wwinpy.parser import from_file
from wwinpy._exceptions import WWINPFormatError

def test_from_file_error(tmp_path):
    """
    Test that the parser raises an error on bad input.
    """
    content = ""  # An empty file

    test_file = tmp_path / "test_bad.wwinp"
    test_file.write_text(content)

    with pytest.raises(WWINPFormatError):
        _ = from_file(str(test_file))
