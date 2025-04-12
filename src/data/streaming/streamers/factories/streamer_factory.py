from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.linear_streamer import LinearStreamer

T = TypeVar("T")


class StreamerFactory(Generic[T], ABC):
    """Interface for streamer factories."""

    @abstractmethod
    def create_streamer(self) -> Optional[LinearStreamer]:
        """
        Returns the next available streamer.

        Returns:
            LinearStreamer: the next streamer instance, or None if no streamers are available
        """
        raise NotImplementedError