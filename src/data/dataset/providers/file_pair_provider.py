from abc import ABC, abstractmethod
from typing import Optional

from src.data.dataclasses.file_pair import FilePair


class FilePairProvider(ABC):
    """An interface for providers of DatasetInstance."""

    @abstractmethod
    def get_next(self) -> Optional[FilePair]:
        """
        Returns the next DatasetInstance instance.

        Returns:
            Optional[FilePair]: a dataset instance
        """
        raise NotImplementedError