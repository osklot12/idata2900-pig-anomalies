from abc import ABC, abstractmethod

from src.data.streaming.streamers.providers.streamer_factory import StreamerFactory


class StreamerProviderFactory(ABC):
    """Interface for streamer provider factories."""

    @abstractmethod
    def create_provider(self) -> StreamerFactory:
        """
        Creates and returns a new StreamerProvider instance.

        Returns:
            StreamerFactory: a new StreamerProvider instance
        """
        raise NotImplementedError