import queue
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.stream import Stream
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.feedables.feedable_queue import FeedableQueue

T = TypeVar("T")


class SequentialStream(Generic[T], Stream[T]):
    """Sequential stream of data, where input streams are ordered sequentially."""

    def __init__(self, buffer_size: int = 3):
        """
        Initializes a SequentialStream instance.

        Args:
            buffer_size (int): the size of the internal buffer
        """
        self._queue_stream: queue.Queue[Optional[queue.Queue[T]]] = queue.Queue(maxsize=buffer_size)
        self._buffer_size = buffer_size

        self._current_stream = None
        self._eos = False

    def read(self) -> Optional[T]:
        result = None

        if not self._eos:
            if self._current_stream is None:
                self._current_stream = self._queue_stream.get()

            while self._current_stream and result is None:
                instance = self._current_stream.get()
                if not instance is None:
                    result = instance
                else:
                    self._current_stream = self._queue_stream.get()
                    if self._current_stream is None:
                        self._eos = True

        return result

    def open_feedable_stream(self, timeout: float = None) -> Feedable[T]:
        """
        Returns the next input to stream to.

        Returns:
            Feedable[T]: the next input

        Raises:
            queue.Full: raised when the internal queue is full after the timeout period
        """
        q = queue.Queue()
        self._queue_stream.put(q, timeout=timeout)
        return FeedableQueue[T](q)

    def close(self) -> None:
        """Closes the stream."""
        self._queue_stream.put(None)