from src.command.command import Command
from src.data.streaming.managers.runnable_streamer_manager import RunnableStreamerManager


class NotifyStoppedStreamerCommand(Command):
    """A command that signals to a manager that a streamer has stopped."""

    def __init__(self, streamer_id: str, manager: RunnableStreamerManager):
        """
        Initializes a StreamerStoppedCommand instance.

        Args:
            streamer_id (str): The id of the streamer.
            manager (RunnableStreamerManager): The manager to notify.
        """
        self._streamer_id = streamer_id
        self._manager = manager

    def execute(self):
        self._manager.streamer_stopped(self._streamer_id)