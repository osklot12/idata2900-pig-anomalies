from abc import ABC, abstractmethod

from src.data.streamers.streamer_status import StreamerStatus


class StreamerStatusTracker(ABC):
    """An interface for classes that tracks streamer statuses."""

    @abstractmethod
    def set_streamer_status(self, streamer_id: str, status: StreamerStatus) -> None:
        """
        Sets the status of the streamer.

        Args:
            streamer_id (str): The ID of the streamer.
            status (StreamerStatus): The status of the streamer.
        """
        raise NotImplementedError