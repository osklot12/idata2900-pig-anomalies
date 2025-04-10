from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Optional

T = TypeVar("T")


class Stream(Generic[T], ABC):
    """A streams of data."""

    @abstractmethod
    def read(self) -> Optional[T]:
        """
        Returns the next item in the streams, blocking if no such item is available.

        Returns:
            T: the item, or None if end of streams is reached
        """
        raise NotImplementedError