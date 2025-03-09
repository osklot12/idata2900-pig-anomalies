from abc import ABC, abstractmethod

class Streamer(ABC):
    """An interface for classes that streams data forward."""

    @abstractmethod
    def stream(self) -> None:
        """Starts streaming data."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops streaming data."""
        pass

    @abstractmethod
    def wait_for_completion(self) -> None:
        """Waist for the end of stream while blocking."""
        raise NotImplementedError