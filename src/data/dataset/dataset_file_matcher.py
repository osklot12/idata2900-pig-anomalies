from abc import ABC, abstractmethod

from src.data.dataset.matching_error_strategy import MatchingErrorStrategy


class DatasetFileMatcher(ABC):
    """An interface for dataset file matchers."""

    @abstractmethod
    def remove_unmatched_file(self, file_name: str) -> None:
        """
        Removes an unmatched file from the matcher.
        This does not remove it from the source.

        Args:
            file_name (str): name of the file to remove
        """
        raise NotImplementedError

    @abstractmethod
    def set_matching_error_strategy(self, strategy: MatchingErrorStrategy) -> None:
        """
        Sets the strategy for errors encountered while matching..

        Args:
            strategy (MatchingErrorStrategy): the strategy to use
        """
        raise NotImplementedError