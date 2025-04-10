from dataclasses import dataclass
from typing import Generic, TypeVar

from src.data.dataset.streams.stream import Stream
from src.data.streaming.managers.streamer_manager import StreamerManager

T = TypeVar("T")

@dataclass(frozen=True)
class ManagedStream(Generic[T]):
    stream: Stream[T]
    manager: StreamerManager

    def start(self) -> None:
        """Starts streaming."""
        self.manager.run()

    def stop(self) -> None:
        """Stops streaming."""
        self.manager.stop()