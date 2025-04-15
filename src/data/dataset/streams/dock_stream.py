import queue
import threading
from typing import TypeVar, Generic, Optional

from src.data.dataset.streams.writable_stream import WritableStream
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.consuming_queue import ConsumingQueue
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

GET_ENTRY_LOOP_TIMEOUT = 0.1

class DockStream(Generic[T], WritableStream[T]):
    """Sequential streams of data, where input streams are ordered sequentially."""

    def __init__(self, buffer_size: int = 3, dock_size: int = 100):
        """
        Initializes a SequentialStream instance.

        Args:
            buffer_size (int): the size of the internal buffer
            dock_size (int): the size of each dock
        """
        self._dock_queue: queue.Queue[Optional[queue.Queue[T]]] = queue.Queue(maxsize=buffer_size)
        self._buffer_size = buffer_size
        self._dock_size = dock_size

        self._current_dock = None
        self._eos = False
        self._closed = AtomicBool(False)

        self._get_entry_lock = threading.Lock()

    def read(self) -> Optional[T]:
        result = None

        if not self._eos:
            if self._current_dock is None:
                self._current_dock = self._dock_queue.get()

            while self._current_dock and result is None:
                instance = self._current_dock.get()

                if instance is not None:
                    result = instance

                else:
                    self._current_dock = self._dock_queue.get()
                    if self._current_dock is None:
                        self._eos = True

        return result

    def get_consumer(self, release: Optional[AtomicBool] = None) -> Optional[Consumer[T]]:
        dock = None

        with self._get_entry_lock:
            keep_trying = True
            while not self._closed and dock is None and keep_trying:
                q = queue.Queue(maxsize=self._dock_size)
                try:
                    self._dock_queue.put(q, timeout=GET_ENTRY_LOOP_TIMEOUT)
                    dock = ConsumingQueue[T](q=q, release=release)
                except queue.Full:
                    pass

                if release is not None and release:
                    keep_trying = False

        return dock

    def close(self) -> None:
        """Closes the streams."""
        if not self._closed:
            self._dock_queue.put(None)
            self._closed.set(True)