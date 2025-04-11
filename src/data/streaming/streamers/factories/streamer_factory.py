from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.streamer import Streamer

T = TypeVar("T")


class StreamerFactory(Generic[T], ABC):
    """Interface for streamer factories."""

    @abstractmethod
    def create_streamer(self, consumer: Consumer[T]) -> Optional[Streamer]:
        """
        Returns the next available streamer.

        Args:
            consumer (Consumer[T]): the consumer of the streaming data

        Returns:
            Streamer: the next streamer instance, or None if no streamers are available
        """
        raise NotImplementedError