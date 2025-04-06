import random
from typing import List, Optional

from src.data.dataset.selectors.string_selector import StringSelector


class RandomStringSelector(StringSelector):
    """Randomly selects a file with a valid suffix from the available list."""

    def __init__(self, strings: List[str]):
        """
        Initializes a RandomStringSelector instance.

        Args:
            strings (List[str]): the list of strings to select from
        """
        self._strings = strings

    def next(self) -> Optional[str]:
        return random.choice(self._strings)