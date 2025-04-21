import queue
from typing import Generic, TypeVar, Optional

from src.data.pipeline.consumer import Consumer
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

CONSUME_LOOP_TIMEOUT = 0.1


class ConsumingQueue(Generic[T], Consumer[T]):
    """Consumer adapter for queues."""

    def __init__(self, q: queue.Queue, release: Optional[AtomicBool] = None):
        """
        Initializes a ConsumingQueue instance.

        Args:
            q (queue.Queue): the queue to feed
            release (Optional[AtomicBool]): optional flag for releasing the block on `consume`
        """
        self._queue = q
        self._release = release

    def consume(self, data: T) -> bool:
        success = False

        keep_trying = True
        while keep_trying and not success:
            try:
                self._queue.put(data, timeout=CONSUME_LOOP_TIMEOUT)
                success = True
            except queue.Full:
                pass

            if self._release is not None and self._release:
                keep_trying = False

        return success

    @property
    def queue(self) -> queue.Queue:
        """
        Returns the queue.

        Returns:
            queue.Queue: the queue
        """
        raise NotImplementedError
