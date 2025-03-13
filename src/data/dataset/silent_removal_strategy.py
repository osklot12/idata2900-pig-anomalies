from typing import Callable

from src.data.dataset.matching_error_strategy import MatchingErrorStrategy


class SilentRemovalStrategy(MatchingErrorStrategy):
    """A matching error strategy that silently removes files on error."""

    def __init__(self, remove_file_callback: Callable[[str], None]):
        """
        Initializes a MatchingErrorStrategy instance.

        Args:
            remove_file_callback (Callable[[str], None]): function to remove file.
        """
        self._remove_file_callback = remove_file_callback

    def handle_no_match(self, file_name: str) -> None:
        self._remove_file_callback(file_name)

    def handle_unknown_file(self, file_name: str) -> None:
        self._remove_file_callback(file_name)