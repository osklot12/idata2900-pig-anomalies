from abc import ABC, abstractmethod

from src.data.streaming.streamers.providers.streamer_provider import StreamerProvider


class StreamerProviderFactory(ABC):
    """Interface for streamer provider factories."""

    @abstractmethod
    def create_provider(self) -> StreamerProvider:
        """
        Creates and returns a new StreamerProvider instance.

        Returns:
            StreamerProvider: a new StreamerProvider instance
        """
        raise NotImplementedError