from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic

# key type for instance
K = TypeVar("K")

# label class type
L = TypeVar("L")


class MetaMaker(Generic[K, L], ABC):
    """Generates metadata for a dataset."""

    @abstractmethod
    def make_metadata(self) -> Dict[K, Dict[L, int]]:
        """
        Creates metadata for this dataset.

        Returns:
            Dict[K, Dict[L, int]]: the metadata
        """
        raise NotImplementedError
