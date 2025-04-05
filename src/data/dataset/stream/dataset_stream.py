from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

T = TypeVar("T")


class DatasetStream(Generic[T], ABC):
    """A stream of data for a dataset."""

    @abstractmethod
    def get_batch(self, n: int) -> Optional[List[T]]:
        """
        Returns a batch of the next n items in the stream, blocking until such a batch is available.

        Args:
            n (int): the batch size

        Returns:
            List[T]: the batch, or None if end of stream is reached
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the stream to its initial state."""
        raise NotImplementedError
