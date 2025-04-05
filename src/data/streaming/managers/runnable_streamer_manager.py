from abc import ABC, abstractmethod


class RunnableStreamerManager(ABC):
    """An interface for streams managers that can be run automatically."""

    @abstractmethod
    def run(self) -> None:
        """Starts the streams manager, managing its streamers automatically."""
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