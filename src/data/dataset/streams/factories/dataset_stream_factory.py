from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.writeable_stream import WriteableStream

T = TypeVar("T")


class DatasetStreamFactory(Generic[T], ABC):
    """Interface for dataset stream factories."""

    @abstractmethod
    def create_train_stream(self) -> WriteableStream[T]:
        """
        Creates a stream for the training set.

        Returns:
            WriteableStream[T]: the training set stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_validation_stream(self) -> WriteableStream[T]:
        """
        Creates a stream for the validation set.

        Returns:
            WriteableStream[T]: the validation set stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_test_stream(self) -> WriteableStream[T]:
        """
        Creates a stream for the test set.

        Returns:
            WriteableStream[T]: the test set stream
        """
        raise NotImplementedError