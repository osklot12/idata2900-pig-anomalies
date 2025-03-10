from src.command.command import Command
from src.data.data_structures.concurrent_dict import ConcurrentDict
from src.data.streamers.streamer_status import StreamerStatus


class UpdateStreamerStatusCommand(Command):
    """Updates the status of a streamer."""

    def __init__(self, streamer_id: str, new_status: StreamerStatus, tracker: ConcurrentDict[str, StreamerStatus]):
        """
        Initializes an instance of UpdateStreamerStatusCommand.

        Args:
            streamer_id (str): The id of the streamer.
            new_status (StreamerStatus): The new status of the streamer.
            tracker (ConcurrentDict[str, StreamerStatus]): The dictionary used for tracking the streamer status.
        """
        self.streamer_id = streamer_id
        self.new_status = new_status
        self.tracker = tracker

    def execute(self):
        self.tracker.set(self.streamer_id, self.new_status)