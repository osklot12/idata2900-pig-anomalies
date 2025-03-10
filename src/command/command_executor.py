from abc import ABC, abstractmethod

from src.command.command import Command


class CommandExecutor(ABC):
    """An interface for classes that executes commands."""

    @abstractmethod
    def submit(self, command: Command) -> None:
        """
        Submits a command for execution.

        Args:
            command (Command): The command to queue.
        """
        raise NotImplementedError()