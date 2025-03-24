from abc import ABC, abstractmethod

from src.data.parsing.string_parser import StringParser


class StringParserFactory(ABC):
    """An interface for StringParser factories."""

    @abstractmethod
    def create_string_parser(self) -> StringParser:
        """
        Creates a new StringParser instance.

        Returns:
            StringParser: a new StringParser instance
        """
        raise NotImplementedError