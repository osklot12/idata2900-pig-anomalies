from abc import ABC, abstractmethod
from typing import List

class SourceRegistry(ABC):
    """An interface for dataset source registries."""

    @abstractmethod
    def get_source_ids(self) -> set[str]:
        """
        Returns a set of the source IDs.

        Returns:
            set[str]: set of source IDs
        """
        raise NotImplementedError