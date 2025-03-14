from abc import ABC, abstractmethod

from src.data.streaming.streamers.streamer import Streamer


class StreamerFactory(ABC):
    """Factory interface for creating streamers."""

    @abstractmethod
    def get_next_streamer(self) -> Streamer:
        """
        Creates and returns the next available streamer.
c
        Returns:
            Streamer: a new streamer instance
        """
        raise NotImplementedError