from src.command.command import Command
from src.data.streaming.streamers import Streamer


class StopStreamerCommand(Command):
    """Command to stop a streamer."""

    def __init__(self, streamer: Streamer):
        """
        Initializes a StopStreamerCommand instance.

        Args:
            streamer (Streamer): the streamer to stop.
        """
        self.streamer = streamer

    def execute(self):
        self.streamer.stop()