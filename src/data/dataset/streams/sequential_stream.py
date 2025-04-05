import queue
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.stream import Stream

T = TypeVar("T")


class SequentialStream(Generic[T], Stream[T]):
    """Sequential streams of data, order of data items is predetermined."""

    def __init__(self, buffer_size: int = 3):
        """Initializes a SequentialStream instance."""
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

    @property
    def queue(self) -> queue.Queue[Optional[queue.Queue[T]]]:
        return self._queue_stream