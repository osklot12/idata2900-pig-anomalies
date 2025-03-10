from abc import ABC, abstractmethod

from src.command.command import Command


class Streamer(ABC):
    """An interface for classes that streams data forward."""

    @abstractmethod
    def stream(self) -> None:
        """Starts streaming data."""
        raise NotImplementedError

    @abstractmethod
    def streaming(self) -> bool:
        """
        Returns True if streamer is currently streaming.

        Returns:
            bool: True if streaming, false otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stops streaming data."""
        pass

    @abstractmethod
    def wait_for_completion(self) -> None:
        """Waist for the end of stream while blocking."""
        raise NotImplementedError

    @abstractmethod
    def add_eos_command(self, command: Command) -> None:
        """
        Adds a command that executes on end of stream.

        Args:
            command (Command): The command to execute.
        """
        raise NotImplementedError