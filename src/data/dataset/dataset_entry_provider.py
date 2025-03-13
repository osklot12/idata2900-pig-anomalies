from abc import ABC, abstractmethod
from typing import Tuple


class DatasetEntryProvider(ABC):
    """An interface for providers of paths of video - annotation pairs in a dataset."""

    def get_random(self) -> Tuple[str, str]:
        """
        Returns video - annotation path pair for a random dataset entry.

        Returns:
            Tuple[str, str]: tuple of video path and annotation path.
        """
        raise NotImplementedError