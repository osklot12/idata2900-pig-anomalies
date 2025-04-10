from abc import ABC, abstractmethod

from src.data.streaming.streamers.providers.streamer_pair_provider import StreamerPairProvider


class StreamerPairProviderFactory(ABC):
    """Interface for streamer pair provider factories."""

    @abstractmethod
    def create_pair_provider(self) -> StreamerPairProvider:
        """
        Creates and return a new StreamerPairProvider instance.

        Returns:
            StreamerPairProvider: the new StreamerPairProvider instance
        """
        raise NotImplementedError
