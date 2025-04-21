from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.streams.writable_stream import WritableStream

T = TypeVar("T")

class WritableStreamFactory(Generic[T], ABC):
    """Interface for writable stream factories."""

    @abstractmethod
    def create_stream(self) -> WritableStream[T]:
        """
        Creates and returns a writable stream instance.

        Returns:
            WritableStream[T]: the created stream instance
        """
        raise NotImplementedError