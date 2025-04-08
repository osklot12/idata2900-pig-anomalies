from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

T = TypeVar("T")


class Feedable(Generic[T], ABC):
    """Interface for feedable objects."""

    @abstractmethod
    def feed(self, food: Optional[T], timeout: Optional[float] = None) -> bool:
        """
        Feeds some food.

        Args:
            food (Optional[T]): the food to feed, or None if there is no more food
            timeout (Optional[float]): the time to wait before returning

        Returns:
            bool: True if successfully fed, False otherwise
        """
        raise NotImplementedError