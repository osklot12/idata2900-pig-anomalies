from abc import ABC, abstractmethod
from typing import Type, List

from src.data.loading.streamer import Streamer


class StreamerProvider(ABC):
    """Provides streamers on request."""

    @abstractmethod
    def get_streamers(self) -> List[Type[Streamer]]:
        """
        Returns a list of streamers.

        Returns:
            List[Type[Streamer]]: List of streamers.
        """
        raise NotImplementedError