from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.managed.managed_stream import ManagedStream

T = TypeVar("T")

class StreamFactory(Generic[T], ABC):
    """Interface for managed streams factories."""

    @abstractmethod
    def create_stream(self) -> ManagedStream[T]:
        """
        Creates a data stream.

        Returns:
            ManagedStream: the created data stream
        """
        raise NotImplementedError