from typing import List, Optional

from src.data.dataset.splitters.factories.splitter_factory import SplitterFactory, T
from src.data.dataset.splitters.splitter import Splitter
from src.data.dataset.splitters.string_set_splitter import StringSetSplitter


class StringSetSplitterFactory(SplitterFactory[str]):
    """Factory for creating StringSetSplitter instances."""

    def __init__(self, weights: List[float], strings: Optional[List[str]] = None, seed: int = 42):
        """
        Initializes a StringSetSplitterFactory instance.

        Args:
            weights (List[float]): weights for each split
            strings (Optional[List[str]]): optional list of strings
        """
        self._weights = weights
        self._strings = strings
        self._seed = seed

    def create_splitter(self) -> Splitter[T]:
        if self._strings:
            splitter = StringSetSplitter(weights=list(self._weights), strings=list(self._strings), seed=self._seed)
        else:
            splitter = StringSetSplitter(weights=list(self._weights), seed=self._seed)

        return splitter