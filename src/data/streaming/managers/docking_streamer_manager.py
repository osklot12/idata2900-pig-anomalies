import queue
import threading
from typing import TypeVar, Generic

from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.dataset.streams.dock_stream import DockStream
from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.streamer import Streamer

T = TypeVar("T")


class DockingStreamerManager(Generic[T], ConcurrentStreamerManager):
    """A streamer manager for directing streamers to a SequentialStream."""

    def __init__(self, streamer_factory: StreamerFactory[T], stream: DockStream[T],
                 max_streamers: int = 10):
        """
        Initializes a RoutingStreamerManager instance.

        Args:
            streamer_factory (StreamerFactory[StreamedAnnotatedFrame]): the factory for creating aggregated streamers
            stream (DockStream): the sequential streams to feed
            max_streamers (int): the maximum number of concurrent streamers
        """
        super().__init__(max_streamers)
        self._streamer_factory = streamer_factory
        self._stream = stream

        self._worker = None

    def _worker_loop(self) -> None:
        """Worker thread function."""
        print(f"[SequentialStreamerManager] Ran worker")
        while self._running:
            try:
                feedable = self._stream.dock(timeout=0.1)
                if feedable:
                    streamer = self._streamer_factory.create_streamer(feedable)
                    if streamer:
                        print(f"[SequentialStreamerManager] Launched streamer...")
                        self._launch_streamer(streamer)
                    else:
                        print(f"[SequentialStreamerManager] End of stream")
                        feedable.feed(None)
                        self._stream.close()

            except queue.Full:
                pass

            except RuntimeError as e:
                print(f"[SequentialStreamerManager] Failed to launch streamer: {e}")

    def _setup(self) -> None:
        print(f"[SequentialStreamerManager] Setting up...")
        self._worker = threading.Thread(target=self._worker_loop)
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