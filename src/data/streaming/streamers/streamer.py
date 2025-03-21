from abc import ABC, abstractmethod

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