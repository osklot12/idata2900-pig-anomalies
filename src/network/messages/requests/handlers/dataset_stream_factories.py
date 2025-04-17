from dataclasses import dataclass
from typing import TypeVar, Generic

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.managed.factories.manged_stream_factory import ManagedStreamFactory

# data type for stream outputs
T = TypeVar("T")


@dataclass(frozen=True)
class DatasetStreamFactories(Generic[T]):
    """
    Collection of factories for creating streams for each dataset split.

    Attributes:
        train_stream (ManagedStreamFactory): factory that creates training set streams
        val_stream (ManagedStreamFactory): factory that creates validation set streams
        test_stream (ManagedStreamFactory): factory that creates test set streams
    """
    train_stream: ManagedStreamFactory[T]
    val_stream: ManagedStreamFactory[T]
    test_stream: ManagedStreamFactory[T]

    def for_split(self, split: DatasetSplit) -> ManagedStreamFactory[T]:
        """
        Returns the factory for the given split.

        Args:
            split (DatasetSplit): the split to get the factory for

        Returns:
            ManagedStreamFactory[T]: the factory for the given split
        """
        return {
            DatasetSplit.TRAIN: self.train_stream,
            DatasetSplit.VAL: self.val_stream,
            DatasetSplit.TEST: self.test_stream
        }.get(split)