from abc import ABC, abstractmethod

from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer


class StreamerFactory(ABC):
    """Factory interface for creating streamers."""

    @abstractmethod
    def get_next_streamer(self) -> ThreadedStreamer:
        """
        Creates and returns the next available streamer.
c
        Returns:
            ThreadedStreamer: a new streamer instance
        """
        raise NotImplementedError