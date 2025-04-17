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
        train_factory (ManagedStreamFactory): factory that creates training set streams
        val_factory (ManagedStreamFactory): factory that creates validation set streams
        test_factory (ManagedStreamFactory): factory that creates test set streams
    """
    train_factory: ManagedStreamFactory[T]
    val_factory: ManagedStreamFactory[T]
    test_factory: ManagedStreamFactory[T]

    def for_split(self, split: DatasetSplit) -> ManagedStreamFactory[T]:
        """
        Returns the factory for the given split.

        Args:
            split (DatasetSplit): the split to get the factory for

        Returns:
            ManagedStreamFactory[T]: the factory for the given split
        """
        return {
            DatasetSplit.TRAIN: self.train_factory,
            DatasetSplit.VAL: self.val_factory,
            DatasetSplit.TEST: self.test_factory
        }.get(split)