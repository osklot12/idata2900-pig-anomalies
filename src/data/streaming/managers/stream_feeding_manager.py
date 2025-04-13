import time
import threading
from typing import TypeVar, Generic

from src.data.dataset.streams.writable_stream import WritableStream
from src.data.pipeline.consumer_provider import ConsumerProvider
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.producer_streamer import ProducerStreamer
from src.data.structures.atomic_bool import AtomicBool

T = TypeVar("T")

WORKER_BACKOFF_TIMEOUT = 0.1


class StreamFeedingManager(Generic[T], ConcurrentStreamerManager):
    """A streamer manager for feeding streams."""

    def __init__(self, streamer_factory: StreamerFactory[T], provider: ConsumerProvider[T], max_streamers: int = 10):
        """
        Initializes a StreamFeedingManager instance.

        Args:
            streamer_factory (StreamerFactory[StreamedAnnotatedFrame]): the factory for creating aggregated streamers
            provider (ConsumerProvider[T]): provider of consumers to consume the stream data
            max_streamers (int): the maximum number of concurrent streamers
        """
        super().__init__(max_streamers)
        self._streamer_factory = streamer_factory
        self._provider = provider

        self._worker = None
        self._shutting_down = AtomicBool(False)

    def _worker_loop(self) -> None:
        """Worker thread function."""
        print(f"[StreamFeedingManager] Ran worker")
        while self._running:
            if self.n_active_streamers() < self._max_streamers:
                try:
                    consumer = self._provider.get_consumer(self._shutting_down)
                    print(f"[StreamFeedingManager] Consumer: {consumer}")
                    if consumer:
                        streamer = self._streamer_factory.create_streamer()
                        print(f"[StreamFeedingManager] Streamer: {streamer}")
                        streamer.connect(consumer)
                        if streamer:
                            print(f"[StreamFeedingManager] Launched streamer...")
                            self._launch_streamer(streamer)
                        else:
                            print(f"[StreamFeedingManager] End of stream")
                            consumer.consume(None)

                except RuntimeError as e:
                    print(f"[StreamFeedingManager] Failed to launch streamer: {e}")
            else:
                time.sleep(WORKER_BACKOFF_TIMEOUT)

    def _setup(self) -> None:
        print(f"[StreamFeedingManager] Setting up...")
        self._worker = threading.Thread(target=self._worker_loop)
        self._worker.start()

    def _run_streamer(self, streamer: ProducerStreamer, streamer_id: str) -> None:
        streamer.wait_for_completion()
        streamer.stop_streaming()

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self._remove_streamer(streamer_id)

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        self._remove_streamer(streamer_id)

    def _stop(self) -> None:
        self._shutting_down.set(True)
        self._worker.join()
        self._worker = None