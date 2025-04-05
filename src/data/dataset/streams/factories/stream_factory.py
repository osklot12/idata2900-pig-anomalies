from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.stream import Stream

T = TypeVar("T")

class StreamFactory(Generic[T], ABC):
    """Interface for streams factories."""

    @abstractmethod
    def create_training_stream(self) -> Stream[T]:
        """
        Creates a data streams for the training set.

        Returns:
            Stream: the training set data streams
        """
        raise NotImplementedError

    @abstractmethod
    def create_validation_stream(self) -> Stream[T]:
        """
        Creates a data streams for the validation set.

        Returns:
            Stream: the validation set data streams
        """
        raise NotImplementedError

    @abstractmethod
    def create_test_stream(self) -> Stream[T]:
        """
        Creates a data streams for the test set.

        Returns:
            Stream: the test set data streams
        """
        raise NotImplementedError