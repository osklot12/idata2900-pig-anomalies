from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic

from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.streamers.streamer import Streamer

T = TypeVar("T")


class StreamerProvider(Generic[T], ABC):
    """Interface for streamer providers."""

    @abstractmethod
    def next_streamer(self, consumer: Feedable[T]) -> Optional[Streamer]:
        """
        Returns the next available streamer.

        Args:
            consumer (Feedable[T]): the consumer of the streaming data

        Returns:
            Streamer: the next streamer instance, or None if no streamers are available
        """
        raise NotImplementedError