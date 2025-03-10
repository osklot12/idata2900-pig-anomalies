import threading
import uuid
from typing import Tuple, Dict, List

from src.command.cleanup_streamer_command import CleanupStreamerCommand
from src.command.command import Command
from src.command.command_executor import CommandExecutor
from src.command.request_execution_command import RequestExecutionCommand
from src.command.update_streamer_status_command import UpdateStreamerStatusCommand
from src.data.streamers.streamer import Streamer
from src.data.streamers.streamer_status import StreamerStatus
from src.data.streamers.streamer_status_tracker import StreamerStatusTracker


class EnsembleStreamer(Streamer, StreamerStatusTracker):
    """A streamer consisting of other streamers."""

    def __init__(self, streamers: Tuple[Streamer, ...]):
        """
        Initializes the StreamerGroup.

        Args:
            streamers (Tuple[Streamer, ...]): The streamers belonging to the group.
        """
        self.streamers: Dict[str, Streamer] = {}
        self.streamer_statuses: Dict[str, StreamerStatus] = {}
        self.streamer_lock = threading.Lock()
        self.status_lock = threading.Lock()
        self.eos_commands: List[Command] = []
        self.executor = CommandExecutor()

        self._init(streamers)

    def _init(self, streamers):
        for streamer in streamers:
            # generate id
            streamer_id = self._generate_streamer_id()

            # add streamer and id to dicts
            self.streamers[streamer_id] = streamer
            self.streamer_statuses[streamer_id] = StreamerStatus.PENDING

            # create eos commands
            status_update_cmd = self._generate_executor_request(
                UpdateStreamerStatusCommand(streamer_id, StreamerStatus.COMPLETED, self)
            )
            cleanup_cmd = self._generate_executor_request(
                CleanupStreamerCommand(streamer)
            )
            streamer.add_eos_command(status_update_cmd)
            streamer.add_eos_command(cleanup_cmd)

    def _generate_executor_request(self, command: Command) -> RequestExecutionCommand:
        """Generates a request command to queue a command at the executor."""
        return RequestExecutionCommand(command, self.executor)

    def set_streamer_status(self, streamer_id: str, status: StreamerStatus):
        with self.status_lock:
            self.streamer_statuses[streamer_id] = status

            # all streamers are done - signal eos
            if self._all_streamers_done():
                self._execute_eos_commands()

    def _execute_eos_commands(self) -> None:
        """Executes all the eos commands."""
        for cmd in self.eos_commands:
            cmd.execute()

    def _all_streamers_done(self) -> bool:
        """Returns True if all streamers have either COMPLETED or FAILED."""
        return all(v in {StreamerStatus.COMPLETED, StreamerStatus.FAILED} for v in self.streamer_statuses.values())

    def stream(self) -> None:
        with self.streamer_lock:
            self.executor.run()
            for streamer in self.streamers.values():
                streamer.stream()

    def stop(self) -> None:
        with self.streamer_lock:
            for streamer in self.streamers.values():
                streamer.stop()
            self.executor.stop()

    def wait_for_completion(self) -> None:
        for streamer in self.streamers.values():
            streamer.wait_for_completion()

        self.executor.stop()

    def add_eos_command(self, command: Command) -> None:
        with self.streamer_lock:
            self.eos_commands.append(command)

    @staticmethod
    def _generate_streamer_id() -> str:
        """
        Generates a unique streamer identifier.
        """
        return str(uuid.uuid4())