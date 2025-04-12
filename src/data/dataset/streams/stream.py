from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

from src.data.pipeline.consumer import Consumer

T = TypeVar("T")


class Stream(Generic[T], ABC):
    """A stream of data."""

    @abstractmethod
    def read(self) -> Optional[T]:
        """
        Returns the next item in the streams, blocking if no such item is available.

        Returns:
            T: the item, or None if end of streams is reached
        """
        raise NotImplementedError

    @abstractmethod
    def get_entry(self) -> Optional[Consumer[T]]:
        """
        Returns entry of input data for the stream.

        Returns:
            Optional[Consumer[T]]: entry of input data for the stream, or None if stream is closed
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes the stream."""
        raise NotImplementedError