from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.stream import Stream

# stream data type
T = TypeVar("T")


class StreamProvider(Generic[T], ABC):
    """Interface for providers of streams."""

    @abstractmethod
    def get_stream(self) -> Stream[T]:
        """
        Returns a Stream instance.

        Returns:
            Stream[T]: a Stream instance
        """
        raise NotImplementedError