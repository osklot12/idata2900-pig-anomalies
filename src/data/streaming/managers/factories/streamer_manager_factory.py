from abc import ABC, abstractmethod

from src.data.streaming.managers.streamer_manager import StreamerManager


class StreamerManagerFactory(ABC):
    """Interface for runnable streamer manager factories."""

    @abstractmethod
    def create_manager(self) -> StreamerManager:
        """
        Creates a StreamerManager instance.

        Return:
            StreamerManager: a streamer manager instance
        """
        raise NotImplementedError