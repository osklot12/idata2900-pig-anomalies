from typing import Callable, Generic, TypeVar, Optional

from src.data.pipeline.consumer import Consumer

T = TypeVar("T")


class FeedableFunc(Generic[T], Consumer[T]):
    """Feedable adapter for callback functions."""

    def __init__(self, callback: Callable[[T], None]):
        """
        Initializes a FeedableFunc instance.

        Args:
            callback (Callable[[T], None]): the callback function to feed
        """
        self._callback = callback

    def consume(self, data: Optional[T]) -> None:
        self._callback(data)