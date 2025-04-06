import random
from typing import List, Optional

from src.data.dataset.selectors.string_selector import StringSelector


class DetermStringSelector(StringSelector):
    """Selects strings in a particular order deterministically, returning each string only once."""

    def __init__(self, files: List[str], seed: int = 42):
        """
        Initializes a DetermStringSelector instance.

        Args:
            files (List[str]): the list of strings to select from
        """
        self._all_files = list(files)
        self._seed = seed
        self._index = 0
        self._shuffled_files = self._shuffle_files()

    def _shuffle_files(self) -> List[str]:
        """Shuffles the strings."""
        shuffled = list(self._all_files)
        rng = random.Random(self._seed)
        rng.shuffle(shuffled)
        return shuffled

    def next(self) -> Optional[str]:
        file = None

        if not self._index >= len(self._shuffled_files):
            file = self._shuffled_files[self._index]
            self._index += 1

        return file

    def reset(self) -> None:
        """Resets the selector to its initial state."""
        self._index = 0
        self._shuffled_files = self._shuffle_files()