from abc import ABC, abstractmethod
from typing import List


class SourceIdProvider(ABC):
    """An interface for providers of source IDs."""

    @abstractmethod
    def get_source_ids(self) -> List[str]:
        """
        Returns a list of source IDs.

        Returns:
            List[str]: list of source IDs
        """
        raise NotImplementedError