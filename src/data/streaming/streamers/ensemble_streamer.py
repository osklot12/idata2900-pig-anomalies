import threading
from typing import Tuple

from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class EnsembleStreamer(StreamerManager, ThreadedStreamer):
    """A streamer consisting of other streamers, abstracting them as one single streamer."""

    def __init__(self, *streamers: Streamer):
        """
        Initializes an instance of EnsembleStreamer.

        Args:
            streamers (Tuple[Streamer, ...]): The streamers belonging to the group.
        """
        ThreadedStreamer.__init__(self)
        StreamerManager.__init__(self)
        self._lock = threading.Lock()
        self._init(streamers)

    def _init(self, streamers):
        """
        Initializes the EnsembleStreamer.

        Args:
            streamers (Tuple[Streamer, ...]): the streamers.
        """
        for streamer in streamers:
            # add streamer
            self._add_streamer(streamer)

    def _stream(self) -> StreamerStatus:
        with self._lock:
            for streamer_id in self.get_streamer_ids():
                self.get_streamer(streamer_id).start_streaming()

        for streamer_id in self.get_streamer_ids():
            self.get_streamer(streamer_id).wait_for_completion()

        return self._get_ensemble_status()

    def _get_ensemble_status(self) -> StreamerStatus:
        """Derives the status of the EnsembleStreamer from all its streamers."""
        statuses = {self.get_streamer(streamer_id).get_status() for streamer_id in self.get_streamer_ids()}

        if StreamerStatus.FAILED in statuses:
            result = StreamerStatus.FAILED
        elif StreamerStatus.STOPPED in statuses:
            result = StreamerStatus.STOPPED
        else:
            result = StreamerStatus.COMPLETED

        return result

    def _stop(self) -> None:
        with self._lock:
            for streamer_id in self.get_streamer_ids():
                self.get_streamer(streamer_id).stop_streaming()