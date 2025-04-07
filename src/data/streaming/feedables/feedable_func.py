from typing import Callable, Generic, TypeVar, Optional

from src.data.streaming.feedables.feedable import Feedable

T = TypeVar("T")


class FeedableFunc(Generic[T], Feedable[Callable[[T], None]]):
    """Feedable adapter for callback functions."""

    def __init__(self, callback: Callable[[T], None]):
        """
        Initializes a FeedableFunc instance.

        Args:
            callback (Callable[[T], None]): the callback function to feed
        """
        self._callback = callback

    def feed(self, food: Optional[T]) -> None:
        self._callback(food)