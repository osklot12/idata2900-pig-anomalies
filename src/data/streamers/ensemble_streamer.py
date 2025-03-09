import threading
from typing import Tuple

from src.data.streamers.streamer import Streamer


class EnsembleStreamer(Streamer):
    """A streamer consisting of other streamers."""

    def __init__(self, streamers: Tuple[Streamer, ...]):
        """
        Initializes the StreamerGroup.

        Args:
            streamers (Tuple[Streamer, ...]): The streamers belonging to the group.
        """
        self.streamers = streamers
        self.lock = threading.Lock()

    def stream(self) -> None:
        for streamer in self.streamers:
            streamer.stream()

    def stop(self) -> None:
        for streamer in self.streamers:
            streamer.stop()