import hashlib
from typing import Iterable, List


class DetermSplitter:
    """Splits string collections into two subsets deterministically."""

    def __init__(self, strings: Iterable[str] = None, threshold: float = 0.5, seed: int = 42):
        """
        Initializes a StringSetSplitter.

        Args:
            strings (Iterable[str]): the collection of strings to split
            threshold (float): the threshold at which to split
            seed (int): the random seed
        """
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")

        self._strings = list(strings)
        self._threshold = threshold
        self._seed = seed

        self._splits = [[], []]

        if strings:
            for string in strings:
                self.add(string)

    def add(self, s: str) -> None:
        """
        Adds a string.

        Args:
            s (str): the string to add
        """
        if not s:
            raise ValueError("string cannot be empty or None")

        self._get_split_for_str(s).append(s)

    def remove(self, s: str) -> None:
        """
        Removes a string.

        Args:
            s (str): the string to remove
        """
        if not s:
            raise ValueError("string cannot be empty or None")

        self._get_split_for_str(s).remove(s)

    def _get_split_for_str(self, s: str) -> List[str]:
        """Returns the appropriate split for the given string."""
        if self._normalized_hash(s) < self._threshold:
            split = self._splits[0]
        else:
            split = self._splits[1]

        return split

    def _normalized_hash(self, s: str) -> float:
        """Gives a normalized hash for a given string."""
        return self._stable_hash(s) / 2 ** 128

    def _stable_hash(self, s: str) -> int:
        """Gives a stable hash for a given string."""
        key = f"{self._seed}_{s}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest(), 16)

    @property
    def first_split(self) -> List[str]:
        """
        Returns the first split.

        Returns:
            List[str]: the first split
        """
        return self._splits[0]

    @property
    def second_split(self) -> List[str]:
        """
        Returns the second split.

        Returns:
            List[str]: the second split
        """
        return self._splits[1]

    def __getitem__(self, item):
        return self._splits[item]

    def __iter__(self):
        return iter(self._splits)