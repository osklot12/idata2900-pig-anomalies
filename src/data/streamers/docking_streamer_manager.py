import threading
import uuid
from typing import Type, Dict

from src.command.command import Command
from src.data.streamers.streamer_manager import StreamerManager
from src.data.streamers.streamer_provider import StreamerProvider
from src.data.streamers.streamer import Streamer


class DockingStreamerManager(StreamerManager):
    """
    A stream manager effective for large sets of finite streams, maintaining a constant number of streams at all time.
    The Streamers "dock" the manager, before leaving and making space for the next streamer.
    """

    def __init__(self, streamer_provider: StreamerProvider, n_streamers: int):
        """
        Initializes a new instance of the DockingStreamManager class.

        Args:
            streamer_provider (Type[StreamerProvider]): Provides streamers.
            n_streamers (int): The number of streamers to maintain at all times.
        """
        super().__init__()
        self.streamer_provider = streamer_provider
        self.n_streamers = n_streamers
        self.lock = threading.Lock()

    def run(self):
        with self.lock:
            self._run_executor()
            # dock streamers until full or until no more streamers are available
            while not self._is_full() and self._dock_next_streamer():
                pass

    def _is_full(self) -> bool:
        """Indicates whether the manager is full of streamers."""
        with self.lock:
            return len(self._get_streamers()) == self.n_streamers

    def stop(self):
        with self.lock:
            for streamer_id in self.get_streamer_ids():
                self.get_streamer(streamer_id).stop()
            self._stop_executor()

    def _dock_next_streamer(self) -> bool:
        """Fetches the next streamer and runs the stream."""
        result = False

        streamer = self.streamer_provider.get_next_streamer()
        if streamer:
            self._add_streamer(streamer)
            streamer.stream()
            result = True

        return result