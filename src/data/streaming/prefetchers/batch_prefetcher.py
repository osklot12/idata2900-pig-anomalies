import queue
import threading
from typing import TypeVar, List

from typing_extensions import Generic

from src.data.dataset.dataset_split import DatasetSplit
from src.data.providers.instance_provider import InstanceProvider
from src.data.streaming.prefetchers.prefetcher import Prefetcher

T = TypeVar("T")


class BatchPrefetcher(Generic[T], Prefetcher[List[T]]):
    """A data batch prefetcher."""

    def __init__(self, instance_provider: InstanceProvider[T], split: DatasetSplit, batch_size: int,
                 buffer_size: int, fetch_timeout: float):
        """
        Initializes a BatchPrefetcher instance.

        Args:
            instance_provider (InstanceProvider): the provider of the data batches
            split (DatasetSplit): the split to sample the batches from
            batch_size (int): the size of the batches
            buffer_size (int): the size of the buffer
            fetch_timeout (float): the time to wait for a response
        """
        if buffer_size < 1:
            raise ValueError("buffer_size must be greater than 0")

        self._provider = instance_provider
        self._split = split
        self._batch_size = batch_size
        self._fetch_timeout = fetch_timeout
        self._queue: queue.Queue[List[T]] = queue.Queue(maxsize=buffer_size)

        self._thread = None
        self._run_lock = threading.Lock()
        self._running_lock = threading.Lock()
        self._running = False

    def run(self) -> None:
        with self._run_lock:
            if self._is_running():
                raise RuntimeError("Prefetcher is already running")
            self._set_running(True)
            self._thread = threading.Thread(target=self._worker)
            self._thread.start()

    def _is_running(self) -> bool:
        """Indicates whether the prefetcher is running or not."""
        with self._running_lock:
            return self._running

    def _set_running(self, status: bool) -> None:
        """Sets the running status of the prefetcher."""
        with self._running_lock:
            self._running = status

    def _worker(self) -> None:
        while self._is_running():
            try:
                batch = self._provider.get_batch(self._split, self._batch_size)
                self._queue.put(batch, timeout=.1)
            except queue.Full:
                pass

    def stop(self) -> None:
        with self._run_lock:
            self._set_running(False)
            self._thread.join()
            self._thread = None

    def get_next_prefetched(self) -> List[T]:
        return self._queue.get(timeout=self._fetch_timeout)