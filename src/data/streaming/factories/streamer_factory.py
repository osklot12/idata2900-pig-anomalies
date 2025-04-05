from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.streamers.streamer import Streamer

T = TypeVar("T")


class StreamerFactory(Generic[T], ABC):
    """Factory interface for creating streamers."""

    @abstractmethod
    def create_streamer(self, consumer: Feedable[T]) -> Optional[Streamer]:
        """
        Creates and returns the next available streamer.

        Args:
            consumer (Feedable[T]): the consumer of the streaming data

        Returns:
            Streamer: a new streamer instance, or None if no streamers are available
        """
        raise NotImplementedError