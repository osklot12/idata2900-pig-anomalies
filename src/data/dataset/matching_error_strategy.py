from abc import ABC, abstractmethod

from src.data.dataset.dataset_file_matcher import DatasetFileMatcher


class MatchingErrorStrategy(ABC):
    """An interface for the strategy pattern for matching errors."""

    def __init__(self, data_provider: DatasetFileMatcher):
        """
        Initializes a MatchingErrorStrategy instance.

        Args:
            data_provider (DatasetFileMatcher): the dataset file matcher
        """
        self._data_provider = data_provider

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