import queue
import threading
from typing import TypeVar, List

from typing_extensions import Generic

from src.data.streaming.prefetchers.prefetcher import Prefetcher
from src.data.structures.atomic_bool import AtomicBool
from src.network.network_dataset_stream import NetworkDatasetStream

T = TypeVar("T")


class BatchPrefetcher(Generic[T], Prefetcher[List[T]]):
    """A data batch prefetcher."""

    def __init__(self, stream: NetworkDatasetStream, batch_size: int,
                 buffer_size: int = 10, fetch_timeout: float = 1.0):
        """
        Initializes a BatchPrefetcher instance.

        Args:
            batch_size (int): the size of the batches
            buffer_size (int): the size of the buffer
            fetch_timeout (float): the time to wait for a response
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than 0")

        self._stream = stream
        self._batch_size = batch_size
        self._fetch_timeout = fetch_timeout
        self._queue: queue.Queue[List[T]] = queue.Queue(maxsize=buffer_size)

        self._thread = None
        self._run_lock = threading.Lock()

        self._running = AtomicBool(False)

    def run(self) -> None:
        with self._run_lock:
            if self._running:
                raise RuntimeError("Prefetcher is already running")

            self._running.set(True)
            self._thread = threading.Thread(target=self._worker)
            self._thread.start()

    def _worker(self) -> None:
        """Worker function that runs on the worker threads."""
        while self._running:
            batch = self._stream.get_batch(self._batch_size)

            put = False
            while not put and self._running:
                try:
                    self._queue.put(batch, timeout=0.1)
                    put = True
                except queue.Full:
                    pass

    def stop(self) -> None:
        with self._run_lock:
            self._running.set(False)
            self._thread.join()
            self._thread = None

    def get(self) -> List[T]:
        return self._queue.get(timeout=self._fetch_timeout)
