import queue
import threading
from typing import TypeVar, List, Optional

from typing_extensions import Generic

from src.data.dataset.streams.stream import Stream
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")


class Prefetcher(Generic[T], Stream[T]):
    """Simple data prefetcher."""

    def __init__(self, stream: Stream, buffer_size: int = 10):
        """
        Initializes a BatchPrefetcher instance.

        Args:
            buffer_size (int): the size of the buffer
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than 0")

        self._stream = stream
        self._queue: queue.Queue[List[T]] = queue.Queue(maxsize=buffer_size)

        self._thread = None
        self._run_lock = threading.Lock()
        self._running = AtomicBool(False)

    def read(self) -> Optional[T]:
        return self._queue.get()

    def run(self) -> None:
        """Runs the prefetcher."""
        with self._run_lock:
            if self._running:
                raise RuntimeError("Prefetcher is already running")

            self._running.set(True)
            self._thread = threading.Thread(target=self._worker)
            self._thread.start()

    def _worker(self) -> None:
        """Worker function that runs on the worker threads."""
        while self._running:
            batch = self._stream.read()
            if batch is None:
                self._running.set(False)

            put = False
            while not put and self._running:
                try:
                    self._queue.put(batch, timeout=0.1)
                    put = True
                except queue.Full:
                    pass

    def stop(self) -> None:
        """Stops the prefetcher."""
        with self._run_lock:
            self._running.set(False)
            self._thread.join()
            self._thread = None