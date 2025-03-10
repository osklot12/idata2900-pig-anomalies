from typing import Callable

from src.command.command import Command


class ConditionalCommand(Command):
    """A command that checks a condition function and executes another command if the condition is met."""

    def __init__(self, command: Command, condition: Callable[[], bool]):
        """
        Initializes an instance of ConditionalCommand.

        Args:
            condition (Callable[[], bool]): A function that returns True when the command should execute.
            command (Command): The command to execute if the condition is met.
        """
        self._condition = condition
        self._command = command

    def execute(self):
        if self._condition():
            self._command.execute()