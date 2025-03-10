from src.command.command import Command
from src.command.command_executor import CommandExecutor


class ConcurrentCommand(Command):
    """
    A command that requests the execution of another command via a CommandExecutor.

    This command acts as a wrapper to submit another command for execution on a separate thread,
    ensuring asynchronous processing through the CommandExecutor.
    """

    def __init__(self, command: Command, executor: CommandExecutor):
        """
        Initializes an instance of RequestExecutionCommand.

        Args:
            command (Command): The command to execute.
            executor (CommandExecutor): The command executor to execute the command.
        """
        self.command = command
        self.executor = executor

    def execute(self):
        self.executor.submit(self.command)