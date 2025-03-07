from abc import ABC, abstractmethod

class Streamer(ABC):
    """An interface for classes that streams data forward."""

    @abstractmethod
    def stream(self):
        """Starts streaming data."""
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        """Stops streaming data."""
        pass

    @abstractmethod
    def wait_for_completion(self):
        """Waist for the end of stream while blocking."""
        raise NotImplementedError