from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.writable_stream import WritableStream

T = TypeVar("T")


class DatasetStreamFactory(Generic[T], ABC):
    """Interface for dataset stream factories."""

    @abstractmethod
    def create_train_stream(self) -> WritableStream[T]:
        """
        Creates a stream for the training set.

        Returns:
            WritableStream[T]: the training set stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_validation_stream(self) -> WritableStream[T]:
        """
        Creates a stream for the validation set.

        Returns:
            WritableStream[T]: the validation set stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_test_stream(self) -> WritableStream[T]:
        """
        Creates a stream for the test set.

        Returns:
            WritableStream[T]: the test set stream
        """
        raise NotImplementedError