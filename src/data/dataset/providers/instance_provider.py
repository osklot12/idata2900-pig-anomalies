from abc import ABC, abstractmethod
from typing import Optional, Iterator

from src.data.dataclasses.dataset_instance import DatasetInstance


class InstanceProvider(ABC):
    """Interface for dataset instance factories."""

    @abstractmethod
    def get(self) -> Optional[DatasetInstance]:
        """
        Returns the next dataset instance.

        Returns:
            DatasetInstance: the next available dataset instance, or None if none are available
        """
        raise NotImplementedError