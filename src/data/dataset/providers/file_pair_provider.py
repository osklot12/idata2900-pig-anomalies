from abc import ABC, abstractmethod
from typing import Optional

from src.data.dataclasses.dataset_instance import DatasetInstance


class FilePairProvider(ABC):
    """An interface for providers of DatasetInstance."""

    @abstractmethod
    def get_next(self) -> Optional[DatasetInstance]:
        """
        Returns the next DatasetInstance instance.

        Returns:
            Optional[DatasetInstance]: a dataset instance
        """
        raise NotImplementedError