from src.data.parsing.factories.string_parser_factory import StringParserFactory
from src.data.parsing.file_base_name_parser import FileBaseNameParser
from src.data.parsing.string_parser import StringParser


class FileBaseNameParserFactory(StringParserFactory):
    """A factory for creating FileBaseNameParser instances."""

    def create_string_parser(self) -> StringParser:
        return FileBaseNameParser()