from abc import abstractmethod

from src.command.command import Command
from src.data.streamers.streamer import Streamer


class StreamManager:
    """An interface for stream managers."""

    @abstractmethod
    def run(self) -> None:
        """Runs the stream manager."""
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops the stream manager."""
        raise NotImplementedError

    @abstractmethod
    def queue_command(self, command: Command) -> None:
        """
        Queue a command for execution.

        Args:
            command (Command): The command to queue.
        """
        raise NotImplementedError