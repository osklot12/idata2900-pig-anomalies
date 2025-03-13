from abc import ABC, abstractmethod


class MatchingErrorStrategy(ABC):
    """An interface for the strategy pattern for matching errors."""

    @abstractmethod
    def handle_unknown_file(self, file_name: str) -> None:
        """
        Handles encountering an unknown file.

        Args:
            file_name (str): the file name
        """
        raise NotImplementedError

    @abstractmethod
    def handle_no_match(self, file_name: str) -> None:
        """
        Handles encountering a file with no match.

        Args:
            file_name (str): the file name
        """
        raise NotImplementedError