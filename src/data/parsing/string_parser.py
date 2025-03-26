from abc import ABC, abstractmethod

class StringParser(ABC):
    """An interface for string parsers."""

    @abstractmethod
    def parse_string(self, string: str) -> str:
        """
        Parses a string.

        Args:
            string (str): the string to parse

        Returns:
            str: the parsed string
        """
        raise NotImplementedError