import queue
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.stream import Stream
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consuming_queue import ConsumingQueue
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")


class DockStream(Generic[T], Stream[T]):
    """Sequential streams of data, where input streams are ordered sequentially."""

    def __init__(self, buffer_size: int = 3, dock_size: int = 100, timeout: Optional[float] = None):
        """
        Initializes a SequentialStream instance.

        Args:
            buffer_size (int): the size of the internal buffer
            dock_size (int): the size of each dock
            timeout (Optional[float]): the timeout in seconds to block on input/output of stream
        """
        self._dock_queue: queue.Queue[Optional[queue.Queue[T]]] = queue.Queue(maxsize=buffer_size)
        self._buffer_size = buffer_size
        self._dock_size = dock_size

        self._timeout = timeout

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

    def get_entry(self) -> Optional[Consumer[T]]:
        dock_input = None

        if not self._closed:
            q = queue.Queue(maxsize=self._dock_size)
            self._dock_queue.put(q, timeout=self._timeout)
            dock_input = ConsumingQueue[T](q)

        return dock_input

    def close(self) -> None:
        """Closes the streams."""
        if not self._closed:
            self._dock_queue.put(None)
            self._closed.set(True)