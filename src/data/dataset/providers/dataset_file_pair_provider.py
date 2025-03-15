from abc import ABC, abstractmethod
from typing import Optional

from src.data.dataclasses.dataset_file_pair import DatasetFilePair


class DatasetFilePairProvider(ABC):
    """An interface for providers DatasetFilePair."""

    @abstractmethod
    def get_file_pair(self) -> Optional[DatasetFilePair]:
        """
        Returns a DatasetFilePair instance.

        Returns:
            Optional[DatasetFilePair]: a dataset file pair
        """
        raise NotImplementedError