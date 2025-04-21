from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.closable_stream import ClosableStream

# stream data type
T = TypeVar("T")


class ClosableStreamFactory(Generic[T], ABC):
    """Interface for closable stream factories."""

    @abstractmethod
    def create_stream(self) -> ClosableStream[T]:
        """
        Creates and returns a stream instance.

        Returns:
            ClosableStream[T]: the created stream instance
        """
        raise NotImplementedError