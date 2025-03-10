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
        self.streamer_provider = streamer_provider
        self.n_streamers = n_streamers
        self.streamers: Dict[str, Streamer] = {}
        self.lock = threading.Lock()

    def run(self):
        # dock streamers until full or until no more streamers are available
        while not self._is_full() and self._dock_next_streamer():
            pass

    def _is_full(self) -> bool:
        """Indicates whether the manager is full of streamers."""
        with self.lock:
            return len(self.streamers) == self.n_streamers

    def stop(self):
        with self.lock:
            for streamer in self.streamers.values():
                streamer.stop()
            self.streamers.clear()

    def terminate_streamer(self, streamer_id: str):
        with self.lock:
            streamer = self.streamers.pop(streamer_id, None)
            if streamer:
                streamer.stop()
            else:
                raise KeyError(f"Streamer ID '{streamer_id}' not found.")

    def queue_command(self, command: Command) -> None:
        pass

    def _dock_next_streamer(self) -> bool:
        """Fetches the next streamer and runs the stream."""
        result = False

        streamer = self.streamer_provider.get_next_streamer()
        if streamer:
            self._add_streamer(streamer)
            streamer.stream()
            result = True

        return result

    @staticmethod
    def _generate_streamer_id() -> str:
        """
        Generates a unique streamer identifier.
        """
        return str(uuid.uuid4())
    def _add_streamer(self, streamer: Streamer):
        """Adds a streamer to the manager."""
        streamer_id = self._generate_streamer_id()
        with self.lock:
            self.streamers[streamer_id] = streamer



