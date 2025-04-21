from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

T = TypeVar("T")


class Consumer(Generic[T], ABC):
    """Interface for consumers of data."""

    @abstractmethod
    def consume(self, data: Optional[T]) -> bool:
        """
        Consumes the given data.

        Args:
            data (T): f the data to be consumed, or None if end ostream
        """
        raise NotImplementedError