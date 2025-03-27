from abc import ABC, abstractmethod
from typing import Optional

from src.data.dataclasses.dataset_instance import DatasetInstance


class DatasetManifest(ABC):
    """An interface for providers of DatasetInstance."""

    @abstractmethod
    def get_dataset_instance(self) -> Optional[DatasetInstance]:
        """
        Returns a DatasetInstance instance.

        Returns:
            Optional[DatasetInstance]: a dataset instance
        """
        raise NotImplementedError