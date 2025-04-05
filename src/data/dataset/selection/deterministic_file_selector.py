import random
from typing import List, Optional

from src.data.dataset.selection.file_selector import FileSelector


class DeterministicFileSelector(FileSelector):
    """Deterministically selects files from a dataset."""

    def __init__(self, files: List[str], seed: int = 42):
        """
        Initializes a DeterministicFileSelector instance.

        Args:
            files (List[str]): the list of file names to select from
        """
        self._all_files = list(files)
        self._seed = seed
        self._index = 0
        self._shuffled_files = self._shuffle_files()

    def _shuffle_files(self) -> List[str]:
        """Shuffles the files."""
        shuffled = list(self._all_files)
        rng = random.Random(self._seed)
        rng.shuffle(shuffled)
        return shuffled

    def select_file(self) -> Optional[str]:
        file = None

        if not self._index >= len(self._shuffled_files):
            file = self._shuffled_files[self._index]
            self._index += 1

        return file

    def reset(self) -> None:
        """Resets the selector to its initial state."""
        self._index = 0
        self._shuffled_files = self._shuffle_files()