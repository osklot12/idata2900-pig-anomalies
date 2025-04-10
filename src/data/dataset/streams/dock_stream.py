import queue
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.stream import Stream
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.feedables.feedable_queue import FeedableQueue
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")


class DockStream(Generic[T], Stream[T]):
    """Sequential streams of data, where input streams are ordered sequentially."""

    def __init__(self, buffer_size: int = 3, dock_size: int = 100):
        """
        Initializes a SequentialStream instance.

        Args:
            buffer_size (int): the size of the internal buffer
        """
        self._dock_queue: queue.Queue[Optional[queue.Queue[T]]] = queue.Queue(maxsize=buffer_size)
        self._buffer_size = buffer_size
        self._dock_size = dock_size

        self._current_dock = None
        self._eos = False
        self._closed = AtomicBool(False)

    def read(self) -> Optional[T]:
        print(f"[SequentialStream] Reading...")
        result = None

        if not self._eos:
            if self._current_dock is None:
                self._current_dock = self._dock_queue.get()

            while self._current_dock and result is None:
                print(f"[SequentialStream] Fetching next instance")
                instance = self._current_dock.get()
                if instance:
                    print(f"[SequentialStream] Got instance")
                    result = instance

                else:
                    self._current_dock = self._dock_queue.get()
                    print(f"[SequentialStream] Fetching next stream")
                    if self._current_dock is None:
                        print("[SequentialStream] End of stream")
                        self._eos = True

        return result

    def dock(self, timeout: float = None) -> Optional[Feedable[T]]:
        """
        Returns the next input to stream to.

        Returns:
            Optional[Feedable[T]]: the opened feedable stream, or None if stream is closed

        Raises:
            queue.Full: raised when the internal queue is full after the timeout period
        """
        dock_input = None

        if not self._closed:
            q = queue.Queue(maxsize=self._dock_size)
            self._dock_queue.put(q, timeout=timeout)
            dock_input = FeedableQueue[T](q)

        return dock_input

    def close(self) -> None:
        """Closes the streams."""
        if not self._closed:
            self._dock_queue.put(None)
            self._closed.set(True)