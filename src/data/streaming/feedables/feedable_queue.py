import queue
from typing import Generic, TypeVar

from src.data.streaming.feedables.feedable import Feedable

T = TypeVar("T")


class FeedableQueue(Generic[T], Feedable[T]):
    """Feedable queue."""

    def __init__(self, q: queue.Queue):
        """
        Initializes a FeedableQueue instance.

        Args:
            q (queue.Queue): the queue to feed
        """
        self._queue = q

    def feed(self, food: T) -> None:
        self._queue.put(food)

    @property
    def queue(self) -> queue.Queue:
        """
        Returns the queue.

        Returns:
            queue.Queue: the queue
        """
        raise NotImplementedError
