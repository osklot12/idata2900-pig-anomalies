from src.command.command import Command
from src.data.data_structures.concurrent_dict import ConcurrentDict
from src.data.streamers.streamer import Streamer


class StopStreamersCommand(Command):
    """A command that stops all streamers in a ConcurrentDict."""

    def __init__(self, streamers: ConcurrentDict[str, Streamer]):
        """
        Initializes a StopStreamersCommand instance.

        Args:
            streamers (ConcurrentDict[str, Streamer]): The streamers to stop.
        """
        self._streamers = streamers

    def execute(self):
        for streamer_id in self._streamers.keys():
            self._streamers.get(streamer_id).stop()