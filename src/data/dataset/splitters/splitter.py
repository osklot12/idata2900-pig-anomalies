from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

T = TypeVar("T")

class Splitter(Generic[T], ABC):
    """Interface for splitters."""

    @abstractmethod
    def add(self, item: T) -> int:
        """
        Adds an item to the splitter.

        Args:
            item (T): the item to add

        Returns:
            int: the index of the split the item was assigned to
        """
        raise NotImplementedError

    @abstractmethod
    def remove(self, item: T) -> None:
        """
        Removes an item from the splitter.

        Args:
            item (T): the item to remove
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def splits(self) -> List[List[T]]:
        """
        Returns the splits.

        Returns:
            List[List[T]]: the splits
        """
        raise NotImplementedError