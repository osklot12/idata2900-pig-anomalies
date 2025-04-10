from abc import ABC, abstractmethod


class StreamerManager(ABC):
    """An interface for streams managers."""

    @abstractmethod
    def run(self) -> None:
        """Starts the streams manager."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops the streamer manager."""
        raise NotImplementedError

    @abstractmethod
    def n_active_streamers(self) -> int:
        """
        Returns the number of active streamers.

        Returns:
            int: Number of active streamers.
        """
        raise NotImplementedError