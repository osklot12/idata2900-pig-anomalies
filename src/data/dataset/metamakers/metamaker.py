from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic

class Metamaker(ABC):
    """Generates metadata for a dataset."""

    @abstractmethod
    def make_metadata(self) -> Dict[int, Dict[str, Dict[str, int]]]:
        """
        Creates metadata for this dataset.

        Returns:
            Dict[int, Dict[str, Dict[str, int]]]: the metadata
        """
        raise NotImplementedError