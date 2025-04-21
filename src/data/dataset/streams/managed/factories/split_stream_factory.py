from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.managed_stream import ManagedStream

T = TypeVar("T")


class SplitStreamFactory(Generic[T], ABC):
    """Interface for factories of dataset split streams."""

    @abstractmethod
    def create_stream(self, split: DatasetSplit) -> ManagedStream[T]:
        """
        Creates a stream for the given split.

        Args:
            split (DatasetSplit): dataset split to create stream for

        Returns:
            ManagedStream[T]: stream for the given split
        """
        raise NotImplementedError
