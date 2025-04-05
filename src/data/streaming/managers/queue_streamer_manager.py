import queue
import threading
from typing import TypeVar, Generic

from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.feedables.feedable_queue import FeedableQueue
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.managers.runnable_streamer_manager import RunnableStreamerManager
from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.streamers.streamer import Streamer
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")


class QueueStreamerManager(Generic[T], ConcurrentStreamerManager):
    """Manages streamers that streams into queues."""

    def __init__(self, streamer_factory: AggregatedStreamerFactory, queue_stream: queue.Queue[queue.Queue[T]],
                 max_streamers: int = 10):
        """
        Initializes a QueueStreamerManager instance.

        Args:
            streamer_factory (AggregatedStreamerFactory): the factory for creating aggregated streamers
            queue_stream (queue.Queue(queue.Queue[T])): a stream of queues to get queues from
            max_streamers (int): the maximum number of concurrent streamers
        """
        super().__init__(max_streamers)
        self._streamer_factory = streamer_factory
        self._queue_stream: queue.Queue[queue.Queue[T]] = queue_stream

        self._worker = None

    def _worker(self) -> None:
        """Worker thread function."""
        while self._running:
            try:
                q = self._queue_stream.get(timeout=0.1)
                streamer = self._streamer_factory.create_streamer(FeedableQueue(q))
                self._launch_streamer(streamer)
            except queue.Empty:
                pass

    def _setup(self) -> None:
        self._worker = threading.Thread(target=self._worker)
        self._worker.start()

    def _run_streamer(self, streamer: Streamer, streamer_id: str) -> None:
        streamer.wait_for_completion()
        streamer.stop_streaming()

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self._remove_streamer(streamer_id)

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        self._remove_streamer(streamer_id)

    def _stop(self) -> None:
        self._worker.join()
        self._worker = None