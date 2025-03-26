from abc import ABC, abstractmethod

from src.data.streaming.streamers.streamer_status import StreamerStatus


class Streamer(ABC):
    """An interface for streamers."""

    @abstractmethod
    def start_streaming(self) -> None:
        """Starts streaming data."""
        raise NotImplementedError

    @abstractmethod
    def stop_streaming(self) -> None:
        """Stops streaming data."""
        raise NotImplementedError

    @abstractmethod
    def wait_for_completion(self) -> None:
        """Blocks while waiting for the streamer to complete."""
        raise NotImplementedError

    @abstractmethod
    def get_status(self) -> StreamerStatus:
        """
        Returns the current status of the streamer.

        Returns:
            StreamerStatus: the current status of streamer
        """
        raise NotImplementedError
