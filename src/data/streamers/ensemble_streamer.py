import threading
from typing import Tuple

from src.command.streamers.stop_streamer_command import StopStreamerCommand
from src.command.concurrent_command import ConcurrentCommand
from src.data.streamers.streamer import Streamer
from src.data.streamers.streamer_manager import StreamerManager
from src.data.streamers.streamer_status import StreamerStatus


class EnsembleStreamer(StreamerManager, Streamer):
    """A streamer consisting of other streamers, abstracting them as one single streamer."""

    def __init__(self, streamers: Tuple[Streamer, ...]):
        """
        Initializes an instance of EnsembleStreamer.

        Args:
            streamers (Tuple[Streamer, ...]): The streamers belonging to the group.
        """
        Streamer.__init__(self)
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

            # cmd for cleaning up the streamer resources immediately after completing
            cleanup_cmd = self._get_cleanup_cmd(streamer)

            streamer.add_eos_command(cleanup_cmd)

    def _stream(self) -> StreamerStatus:
        with self._lock:
            self._run_executor()
            for streamer_id in self.get_streamer_ids():
                self.get_streamer(streamer_id).stream()

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
                self.get_streamer(streamer_id).stop()

            self._stop_executor()

    def _get_cleanup_cmd(self, streamer):
        """Returns a command that cleans up streamer resources."""
        return ConcurrentCommand(
            command=StopStreamerCommand(streamer),
            executor=self._get_executor()
        )