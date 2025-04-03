import queue
import threading
import time
from typing import TypeVar, List

from typing_extensions import Generic

from src.data.dataset.dataset_split import DatasetSplit
from src.data.providers.batch_provider import BatchProvider
from src.data.streaming.prefetchers.prefetcher import Prefetcher
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")


class BatchPrefetcher(Generic[T], Prefetcher[List[T]]):
    """A data batch prefetcher."""

    def __init__(self, batch_provider: BatchProvider[T], split: DatasetSplit, batch_size: int,
                 buffer_size: int = 10, fetch_timeout: float = 1.0):
        """
        Initializes a BatchPrefetcher instance.

        Args:
            batch_provider (BatchProvider): the provider of the data batches
            split (DatasetSplit): the split to sample the batches from
            batch_size (int): the size of the batches
            buffer_size (int): the size of the buffer
            fetch_timeout (float): the time to wait for a response
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than 0")

        self._provider = batch_provider
        self._split = split
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
            time.sleep(.2)
            try:
                batch = self._provider.get_batch(self._split, self._batch_size)
                self._queue.put(batch, timeout=.1)
            except queue.Full:
                pass

    def stop(self) -> None:
        with self._run_lock:
            self._running.set(False)
            self._thread.join()
            self._thread = None

    def get(self) -> List[T]:
        return self._queue.get(timeout=self._fetch_timeout)