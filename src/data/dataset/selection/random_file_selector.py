import random
from typing import List, Optional

from src.data.dataset.selection.file_selector import FileSelector


class RandomFileSelector(FileSelector):
    """Randomly selects a file with a valid suffix from the available list."""

    def __init__(self, suffixes: List[str]):
        """
        Initializes a RandomFileSelector instance.

        Args:
            suffixes (List[str]): list of valid file suffixes for the selected file
        """
        self._suffixes = suffixes

    def select_file(self, candidates: List[str]) -> Optional[str]:
        selected_file = None

        valid_files = [file for file in candidates if self._has_valid_suffix(file)]
        if valid_files:
            selected_file = random.choice(valid_files)

        return selected_file

    def _has_valid_suffix(self, file_name: str) -> bool:
        """Checks if the file name has a valid suffix."""
        return any(file_name.endswith(suffix) for suffix in self._suffixes)