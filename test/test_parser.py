# test_parser.py

import pytest
from wwpy.parser import from_file
from wwpy._exceptions import WWINPFormatError

def test_from_file_minimal(tmp_path):
    """
    Test parsing a very small or minimal WWINP file.
    """
    # Create a temporary file to mimic a WWINP input
    content = """\
1 2 1 10 PROBID123
5
10
1.0 1.0 1.0 0.0 0.0 0.0
2.0 3.0 4.0 5.0
0.0  1.0  2.0
"""  
    # This content is not necessarily correct for your file format
    # It's just a placeholder. You should adapt it to a minimal valid example.

    test_file = tmp_path / "test.wwinp"
    test_file.write_text(content)

    # Now parse it
    try:
        data = from_file(str(test_file))
    except WWINPFormatError as e:
        pytest.fail(f"Parsing raised an unexpected WWINPFormatError: {e}")

    # Perform asserts on the data structure
    assert data.header.if_ == 1
    assert data.header.iv == 2
    assert data.header.probid == "PROBID123"
    # etc.

def test_from_file_error(tmp_path):
    """
    Test that the parser raises an error on bad input.
    """
    content = ""  # An empty file

    test_file = tmp_path / "test_bad.wwinp"
    test_file.write_text(content)

    with pytest.raises(WWINPFormatError):
        _ = from_file(str(test_file))
