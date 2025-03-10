import queue

from src.command.command import Command


class QueueCommand(Command):
    """A command that executes a queue of other commands."""

    def __init__(self, commands: queue.Queue[Command]):
        """
        Initializes an instance of QueueCommand.

        Args:
            commands (queue.Queue): The queued commands to execute.
        """
        self._commands = commands

    def execute(self):
        while not self._commands.empty():
            cmd = self._commands.get()
            cmd.execute()