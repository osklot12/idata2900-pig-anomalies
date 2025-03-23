from abc import ABC, abstractmethod
from typing import List

class DatasetSource(ABC):
    """An interface for dataset sources."""

    @abstractmethod
    def get_source_ids(self) -> set[str]:
        """
        Returns a set of the source IDs.

        Returns:
            set[str]: set of source IDs
        """
        raise NotImplementedError