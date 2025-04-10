from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.stream import Stream

T = TypeVar("T")

class StreamFactory(Generic[T], ABC):
    """Interface for streams factories."""

    @abstractmethod
    def create__stream(self) -> Stream[T]:
        """
        Creates a data stream.

        Returns:
            Stream: the created data stream
        """
        raise NotImplementedError