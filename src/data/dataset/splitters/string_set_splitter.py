import hashlib
from typing import Iterable, List, Optional

from src.data.dataset.splitters.splitter import Splitter


class StringSetSplitter(Splitter[str]):
    """Splits string collections into subsets deterministically."""

    def __init__(self, weights: List[float], strings: Optional[Iterable[str]] = None, seed: int = 42):
        """
        Initializes a Splitter instance.

        Args:
            weights (List[float]): list of weights for each split (must sum to 1), defaults to [0.5, 0.5]
            strings (Iterable[str]): the collection of strings to split
            seed (int): the random seed for hashing
        """
        if weights is None:
            weights = [0.5, 0.5]

        if len(weights) < 2:
            raise ValueError("At least two weights must be provided")

        if not abs(sum(weights) - 1.0) < 1e-6:
            raise ValueError("Weights must sum to 1.0")

        self._strings = list(strings) if strings else []
        self._weights = weights
        self._seed = seed
        self._thresholds = self._compute_thresholds(weights)
        self._splits = [[] for _ in weights]

        for s in self._strings:
            self.add(s)

    @staticmethod
    def _compute_thresholds(weights: List[float]) -> List[float]:
        """Compute the cumulative threshold from weights."""
        thresholds = []
        cumulative = 0.0
        for w in weights:
            cumulative += w
            thresholds.append(cumulative)
        return thresholds

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

    def _get_split_index(self, s: str) -> int:
        """Determine the split index based on normalized hash and thresholds."""
        index = -1

        h = self._normalized_hash(s)
        i = 0
        while i < len(self._thresholds) and index == -1:
            if h < self._thresholds[i]:
                index = i
            i += 1

        if index == -1:
            index = len(self._thresholds) - 1

        return index

    def _get_split_for_str(self, s: str) -> List[str]:
        """Returns the appropriate split for the given string."""
        return self._splits[self._get_split_index(s)]

    def _normalized_hash(self, s: str) -> float:
        """Gives a normalized hash for a given string."""
        return self._stable_hash(s) / 2 ** 128

    def _stable_hash(self, s: str) -> int:
        """Gives a stable hash for a given string."""
        key = f"{self._seed}_{s}".encode("utf-8")
        return int(hashlib.md5(key).hexdigest(), 16)

    @property
    def splits(self) -> List[List[str]]:
        """
        Returns the splits.

        Returns:
            List[List[str]]: the splits
        """
        return list(self._splits)

    def __getitem__(self, item):
        return self._splits[item]

    def __iter__(self):
        return iter(self._splits)