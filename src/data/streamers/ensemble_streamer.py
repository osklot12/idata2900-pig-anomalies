import threading
from typing import Tuple

from src.command.command import Command
from src.data.streamers.streamer import Streamer


class EnsembleStreamer(Streamer):
    """A streamer consisting of other streamers."""

    def __init__(self, streamers: Tuple[Streamer, ...], termination_command: Command = None):
        """
        Initializes the StreamerGroup.

        Args:
            streamers (Tuple[Streamer, ...]): The streamers belonging to the group.
            termination_command (Command): A optional termination command, executed when all streamers are finished.
        """
        self.streamers = streamers
        self.termination_command = termination_command
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def stream(self):
        with self._lock:
            self._check_thread_safety()

            for streamer in self.streamers:
                streamer.stream()

            self._thread = threading.Thread(target=self._wait_worker)
            self._thread.start()

    def _wait_worker(self):
        for stream in self.streamers:
            stream.wait_for_completion()

        if self.termination_command:
            self.termination_command.execute()

    def stop(self):
        with self._lock:
            for stream in self.streamers:
                stream.stop()
        self._join_thread()

    def wait_for_completion(self):
        self._join_thread()

    def _check_thread_safety(self):
        """Ensures only one instance runs at a time."""
        if self._thread and self._thread.is_alive():
            raise RuntimeError("Streamers already running on another thread.")

    def _join_thread(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()
            self._thread = None