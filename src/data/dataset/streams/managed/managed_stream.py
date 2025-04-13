from typing import Optional, Generic, TypeVar

from src.data.dataset.streams.stream import Stream
from src.data.streaming.managers.streamer_manager import StreamerManager

T = TypeVar("T")

class ManagedStream(Generic[T], Stream[T], StreamerManager):
    """Stream managed by some stream manager."""

    def __init__(self, stream: Stream[T], manager: StreamerManager):
        """
        Initializes a ManagedStream instance.

        Args:
            stream (Stream[T]): the stream to read from
            manager (StreamerManager[T]): the stream manager
        """
        self._stream = stream
        self._manager = manager

    def read(self) -> Optional[T]:
        return self._stream.read()

    def run(self) -> None:
        self._manager.run()

    def stop(self) -> None:
        self._manager.stop()

    def n_active_streamers(self) -> int:
        return self._manager.n_active_streamers()