from abc import ABC, abstractmethod
from typing import List, TypeVar, Generic

from src.data.dataset.dataset_split import DatasetSplit

T = TypeVar('T')

class BatchProvider(ABC, Generic[T]):
    """Interface for data batch providers."""

    @abstractmethod
    def get_batch(self, split: DatasetSplit, batch_size: int) -> List[T]:
        """
        Returns a batch of data instances.

        Args:
            split (DatasetSplit): the split to sample from
            batch_size (int): the batch size

        Returns:
            List[T]: the batch of data instances
        """
        raise NotImplementedError