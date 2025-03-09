from src.command.command import Command
from src.data.streamers.stream_manager import StreamManager


class TerminateStreamerCommand(Command):
    """A command pattern object that terminates streamers."""

    def __init__(self, streamer_id: str, manager: StreamManager):
        """
        Initializes a new TerminateStreamerCommand object.

        Args:
            streamer_id (str): The id of the streamer to terminate.
            manager (StreamManager): The manager responsible for the streamer.
        """
        self.streamer_id = streamer_id
        self.manager = manager

    def execute(self):
        self.manager.terminate_streamer(self.streamer_id)