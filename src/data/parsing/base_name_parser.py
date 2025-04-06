import os

from src.data.parsing.string_parser import StringParser


class BaseNameParser(StringParser):
    """Strips file directories down to the file base name."""

    def parse_string(self, string: str) -> str:
        if string is None:
            raise ValueError("string cannot be None")

        return os.path.splitext(os.path.basename(string))[0]