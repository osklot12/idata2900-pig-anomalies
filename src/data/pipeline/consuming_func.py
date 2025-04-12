from typing import Callable, Generic, TypeVar, Optional

from src.data.pipeline.consumer import Consumer

T = TypeVar("T")


class ConsumingFunc(Generic[T], Consumer[T]):
    """Consumer adapter for callback functions."""

    def __init__(self, callback: Callable[[T], bool]):
        """
        Initializes a ConsumingFunc instance.

        Args:
            callback (Callable[[T], bool]): the callback function to feed
        """
        self._callback = callback

    def consume(self, data: Optional[T]) -> bool:
        return self._callback(data)