import queue
from typing import Generic, TypeVar

from src.data.pipeline.consumer import Consumer

T = TypeVar("T")


class FeedableQueue(Generic[T], Consumer[T]):
    """Feedable queue."""

    def __init__(self, q: queue.Queue):
        """
        Initializes a FeedableQueue instance.

        Args:
            q (queue.Queue): the queue to feed
        """
        self._queue = q

    def consume(self, data: T) -> None:
        self._queue.put(data)

    @property
    def queue(self) -> queue.Queue:
        """
        Returns the queue.

        Returns:
            queue.Queue: the queue
        """
        raise NotImplementedError
