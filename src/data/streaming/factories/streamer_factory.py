from abc import ABC, abstractmethod

from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer


class StreamerFactory(ABC):
    """Factory interface for creating streamers."""

    @abstractmethod
    def create_streamer(self) -> Streamer:
        """
        Creates and returns the next available streamer.

        Returns:
            Streamer: a new streamer instance
        """
        raise NotImplementedError