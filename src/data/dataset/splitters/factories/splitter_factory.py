from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.data.dataset.splitters.splitter import Splitter

T = TypeVar("T")


class SplitterFactory(Generic[T], ABC):
    """Interface for splitter factories."""

    @abstractmethod
    def create_splitter(self) -> Splitter[T]:
        """
        Creates and returns a Splitter instance.

        Returns:
            Splitter[T]: the Splitter instance
        """
        raise NotImplementedError