import random
from typing import List, Optional

from src.data.dataset.selection.file_selection_strategy import FileSelectionStrategy


class RandomFileSelectionStrategy(FileSelectionStrategy):
    """Randomly selects a file with a valid suffix from the available list."""

    def __init__(self, suffixes: List[str]):
        """
        Initializes a RandomFileSelectionStrategy instance.

        Args:
            suffixes (List[str]): list of valid file suffixes for the selected file
        """
        self._suffixes = suffixes

    def select_file(self, candidates: List[str]) -> Optional[str]:
        valid_files = [file for file in candidates if self._has_valid_suffix(file)]
        return random.choice(valid_files) if valid_files else None

    def _has_valid_suffix(self, file_name: str) -> bool:
        """Checks if the file name has a valid suffix."""
        return any(file_name.endswith(suffix) for suffix in self._suffixes)