from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional

T = TypeVar("T")


class Feedable(Generic[T], ABC):
    """Interface for feedable objects."""

    @abstractmethod
    def feed(self, food: Optional[T]) -> None:
        """
        Feeds some food.

        Args:
            food (T): the food to feed, or None if there is no more food
        """
        raise NotImplementedError