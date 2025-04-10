from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.stream import Stream

T = TypeVar("T")


class DatasetStreamFactory(Generic[T], ABC):
    """Interface for dataset stream factories."""

    @abstractmethod
    def create_train_stream(self) -> Stream[T]:
        """
        Creates a stream for the training set.

        Returns:
            Stream[T]: the training set stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_validation_stream(self) -> Stream[T]:
        """
        Creates a stream for the validation set.

        Returns:
            Stream[T]: the validation set stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_test_stream(self) -> Stream[T]:
        """
        Creates a stream for the test set.

        Returns:
            Stream[T]: the test set stream
        """
        raise NotImplementedError