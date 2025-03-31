from abc import ABC, abstractmethod
from typing import TypeVar, Generic

T = TypeVar("T")

class Prefetcher(ABC, Generic[T]):
    """Interface for data prefetchers."""

    @abstractmethod
    def run(self) -> None:
        """Runs the prefetcher."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops the prefetcher."""
        raise NotImplementedError

    @abstractmethod
    def get(self) -> T:
        """
        Returns the next prefetched data.

        Returns:
            T: the next prefetched data
        """
        raise NotImplementedError