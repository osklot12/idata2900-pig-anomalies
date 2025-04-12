from abc import ABC, abstractmethod
from typing import List, Optional, TypeVar, Generic

T = TypeVar("T")

class Selector(Generic[T], ABC):
    """Strategy for selecting items."""

    @abstractmethod
    def select(self) -> Optional[T]:
        """
        Selects the next item.

        Returns:
            Optional[T]: the selected item, or None if no items are available
        """
        raise NotImplementedError