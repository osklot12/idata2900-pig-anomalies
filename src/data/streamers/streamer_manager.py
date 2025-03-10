import queue
from abc import abstractmethod
from typing import List

from src.command.command import Command
from src.command.command_executor import CommandExecutor
from src.command.concurrent_command_executor import ConcurrentCommandExecutor
from src.data.data_structures.concurrent_dict import ConcurrentDict
from src.data.streamers.streamer import Streamer
from src.data.streamers.streamer_status import StreamerStatus


class StreamerManager:
    """
    A base class for stream managers.
    The StreamerManager provides a mechanism for streamers to manage themselves by submitting commands.
    """

    def __init__(self):
        """Initializes an instance of StreamerManager."""
        self._streamers = ConcurrentDict[str, Streamer]()
        self._streamer_statuses = ConcurrentDict[str, StreamerStatus]()
        self._running = False
        self._executor = ConcurrentCommandExecutor()

    def _run_executor(self) -> None:
        """Runs the command executor."""
        self._executor.run()

    def _get_executor(self) -> CommandExecutor:
        """
        Returns the command executor.

        Returns:
            CommandExecutor: The command executor.
        """
        return self._executor

    def _stop_executor(self) -> None:
        """Stops the command executor."""
        self._executor.stop()

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

    def get_streamer_status(self, streamer_id: str) -> StreamerStatus:
        """
        Returns the status of the streamer with the given id.

        Returns:
            StreamerStatus: The status of the streamer with the given id.
        """
        return self._streamer_statuses.get(streamer_id)

    def _get_streamers(self) -> ConcurrentDict[str, Streamer]:
        """Returns the streamer dictionary."""
        return self._streamers

    def _get_streamer_statuses(self) -> ConcurrentDict[str, StreamerStatus]:
        """Returns the streamer status dictionary."""
        return self._streamer_statuses

    def _add_streamer(self, streamer_id: str, streamer: Streamer) -> None:
        """
        Adds a streamer to the manager.

        Args:
            streamer_id: The streamer id.
            streamer: The streamer to be added.
        """
        self._streamers.set(streamer_id, streamer)

    def _set_streamer_status(self, streamer_id: str, streamer: StreamerStatus) -> None:
        """
        Sets the status of the streamer with the given id.

        Args:
            streamer_id: The streamer id.
            streamer: The status to set.
        """
        self._streamer_statuses.set(streamer_id, streamer)

    def running(self) -> bool:
        """
        Indicates whether the stream manager is running.

        Returns:
            bool: True if the stream manager is running, false otherwise.
        """
        return self._running