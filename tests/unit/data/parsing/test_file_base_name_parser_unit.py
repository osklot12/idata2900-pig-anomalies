import pytest

from src.data.parsing.base_name_parser import BaseNameParser


@pytest.fixture
def parser():
    """Fixture to provide a FileBaseNameParser instance."""
    return BaseNameParser()

@pytest.mark.unit
@pytest.mark.parametrize("input_path,expected", [
    ("video1.mp4", "video1"),
    ("some/path/to/video2.avi", "video2"),
    ("/absolute/path/to/video3.mov", "video3"),
    ("folder/video4", "video4"),
    ("another.folder/video5.mp4", "video5")
])
def test_parse_string(parser, input_path, expected):
    """Tests that parse_string() correctly extracts the file base name."""
    # act
    parsed_string = parser.parse_string(input_path)

    # assert
    assert parsed_string == expected
