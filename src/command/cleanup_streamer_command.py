from src.command.command import Command
from src.data.streamers.streamer import Streamer


class CleanupStreamerCommand(Command):
    """Command to clean up streamer resources."""

    def __init__(self, streamer: Streamer):
        self.streamer = streamer

    def execute(self):
        self.streamer.stop()