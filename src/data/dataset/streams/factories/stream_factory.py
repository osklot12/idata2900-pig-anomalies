from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.stream import Stream

T = TypeVar("T")

class StreamFactory(Generic[T], ABC):
    """Interface for stream factories."""

    @abstractmethod
    def create_stream(self) -> Stream[T]:
        """
        Creates and returns a stream instance.

        Returns:
            Stream[T]: the created stream instance
        """
        raise NotImplementedError