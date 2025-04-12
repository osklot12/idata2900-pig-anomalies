import traceback
from typing import TypeVar, Generic, Optional

from src.data.pipeline.consumer import Consumer
from src.data.streaming.managers.concurrent_streamer_manager import ConcurrentStreamerManager
from src.data.streaming.streamers.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.linear_streamer import LinearStreamer

T = TypeVar("T")


class StaticStreamerManager(Generic[T], ConcurrentStreamerManager):
    """Maintains a static number of streamers, streaming into a single consumer."""

    def __init__(self, streamer_factory: StreamerFactory[T], consumer: Consumer[T], n_streamers: int = 4):
        """
        Initializes a StaticStreamerManager instance.

        Args:
            streamer_factory (StreamerFactory[T]): factory for creating streamers
            consumer (Consumer[T]): the consumer of the streaming data
            n_streamers (int): number of streamers to maintain, defaults to 4
        """
        if n_streamers < 1:
            raise ValueError("n_streamers must be greater than 0")

        super().__init__(n_streamers)
        self._factory = streamer_factory
        self._consumer = consumer
        self._n_streamers = n_streamers

    def _setup(self) -> None:
        for _ in range(self._n_streamers):
            streamer = self._get_next_streamer()
            if streamer:
                self._launch_streamer(streamer)

    def _run_streamer(self, streamer: LinearStreamer, streamer_id: str) -> None:
        streamer.wait_for_completion()
        streamer.stop_streaming()

    def _handle_done_streamer(self, streamer_id: str) -> None:
        self._remove_streamer(streamer_id)
        streamer = self._get_next_streamer()
        if streamer:
            self._launch_streamer(streamer)

    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        traceback.print_exc()
        self._remove_streamer(streamer_id)
        streamer = self._get_next_streamer()
        if streamer:
            self._launch_streamer(streamer)

    def _get_next_streamer(self) -> Optional[LinearStreamer]:
        """Returns the next streamer, or None if no streamer is available."""
        streamer = self._factory.create_streamer()
        streamer.connect(self._consumer)
        return streamer