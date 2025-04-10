from src.data.parsing.factories.string_parser_factory import StringParserFactory
from src.data.parsing.base_name_parser import BaseNameParser
from src.data.parsing.string_parser import StringParser


class BaseNameParserFactory(StringParserFactory):
    """A factory for creating FileBaseNameParser instances."""

    def create_string_parser(self) -> StringParser:
        return BaseNameParser()