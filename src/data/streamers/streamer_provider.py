from abc import ABC, abstractmethod
from typing import Type, List

from src.data.streamers.streamer import Streamer


class StreamerProvider(ABC):
    """Provides streamers on request."""

    @abstractmethod
    def get_next_streamer(self) -> Streamer:
        """
        Returns the next available streamer.

        Returns:
            Streamer: The next available streamer.
        """
        raise NotImplementedError