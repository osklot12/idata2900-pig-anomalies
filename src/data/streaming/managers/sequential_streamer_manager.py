import threading
from typing import TypeVar, Generic

from src.data.dataset.streams.sequential_stream import SequentialStream
from src.data.streaming.factories.aggregated_streamer_factory import AggregatedStreamerFactory
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.streamer import Streamer

T = TypeVar("T")


class SequentialStreamerManager(Generic[T], ConcurrentStreamerManager):
    """A streamer manager for directing streamers to a SequentialStream."""

    def __init__(self, streamer_factory: AggregatedStreamerFactory, stream: SequentialStream, max_streamers: int = 10):
        """
        Initializes a RoutingStreamerManager instance.

        Args:
            streamer_factory (AggregatedStreamerFactory): the factory for creating aggregated streamers
            stream (SequentialStream): the sequential streams to feed
            max_streamers (int): the maximum number of concurrent streamers
        """
        super().__init__(max_streamers)
        self._streamer_factory = streamer_factory
        self._stream = stream

        self._worker = None

    def _worker(self) -> None:
        """Worker thread function."""
        while self._running:
            try:
                feedable = self._stream.open_feedable_stream(timeout=0.1)
                streamer = self._streamer_factory.create_streamer(feedable)
                if streamer:
                    self._launch_streamer(streamer)
                else:
                    self._stream.close()
            except RuntimeError:
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