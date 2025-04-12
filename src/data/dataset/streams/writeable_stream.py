from abc import abstractmethod
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.stream import Stream
from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")


class WriteableStream(Generic[T], Stream[T]):
    """Interface for streams what can be written to."""

    @abstractmethod
    def get_entry(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[T]]:
        """
        Returns entry to write data to the stream.

        Args:
            release (Optional[AtomicBool]): optional flag that cancels the operation when true

        Returns:
            Optional[Consumer[T]]: entry of input data for the stream, or None if stream is closed
        """
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Closes the stream."""
        raise NotImplementedError
