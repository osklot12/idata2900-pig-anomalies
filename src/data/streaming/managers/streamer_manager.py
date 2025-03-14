import uuid
from typing import List

from src.data.data_structures.concurrent_dict import ConcurrentDict
from src.data.streaming.streamers.streamer import Streamer


class StreamerManager:
    """A base class for streamer managers."""

    def __init__(self):
        """Initializes an instance of StreamerManager."""
        self._streamers = ConcurrentDict[str, Streamer]()

    def get_streamer_ids(self) -> List[str]:
        """
        Returns the streamer ids.

        Returns:
            List[str]: The streamer ids.
        """
        return self._streamers.keys()

    def get_streamer(self, streamer_id: str) -> Streamer:
        """
        Returns the streamer with the given id.

        Returns:
            Streamer: The streamer with the given id.
        """
        return self._streamers.get(streamer_id)

    def has_streamer(self, streamer_id: str) -> bool:
        """
        Returns True if the streamer with the given id exists.

        Returns:
            bool: True if streamer exists, false otherwise.
        """
        return self._streamers.contains(streamer_id)

    def _get_streamers(self) -> ConcurrentDict[str, Streamer]:
        """Returns the streamer dictionary."""
        return self._streamers

    def _clear_streamers(self) -> None:
        """Clears the streamers."""
        self._streamers.clear()

    def _add_streamer(self, streamer: Streamer) -> str:
        """
        Adds a streamer to the manager, creating a unique id for it.

        Args:
            streamer: The streamer to be added.

        Returns:
            str: The id of the added streamer.
        """
        unique_id = str(uuid.uuid4())
        self._streamers.set(unique_id, streamer)
        return unique_id

    def _remove_streamer(self, streamer_id: str) -> None:
        """Removes the streamer with the given id."""
        self._streamers.remove(streamer_id)

    @staticmethod
    def _generate_streamer_id() -> str:
        """
        Generates a unique streamer identifier.
        """
        return str(uuid.uuid4())
