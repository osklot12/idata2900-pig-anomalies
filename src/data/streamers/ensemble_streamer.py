import queue
import threading
import uuid
from typing import Tuple

from src.command.cleanup_streamer_command import CleanupStreamerCommand
from src.command.command import Command
from src.command.queue_command import QueueCommand
from src.command.concurrent_command import ConcurrentCommand
from src.command.conditional_command import ConditionalCommand
from src.command.update_streamer_status_command import UpdateStreamerStatusCommand
from src.data.streamers.streamer import Streamer
from src.data.streamers.streamer_manager import StreamerManager
from src.data.streamers.streamer_status import StreamerStatus


class EnsembleStreamer(Streamer, StreamerManager):
    """A streamer consisting of other streamers, abstracting them as one single streamer."""

    def __init__(self, streamers: Tuple[Streamer, ...]):
        """
        Initializes an instance of EnsembleStreamer.

        Args:
            streamers (Tuple[Streamer, ...]): The streamers belonging to the group.
        """
        super().__init__()
        self._lock = threading.Lock()
        self._eos_commands = queue.Queue()
        self._init(streamers)

    def _init(self, streamers):
        """
        Initializes the EnsembleStreamer.

        Args:
            streamers (Tuple[Streamer, ...]): the streamers.
        """
        for streamer in streamers:
            # generate id
            streamer_id = self._generate_streamer_id()

            # add streamer and id to dicts
            self._add_streamer(streamer_id, streamer)
            self._set_streamer_status(streamer_id, StreamerStatus.PENDING)

            # cmd for updating streamer status once finished
            update_status_cmd = self._get_update_status_cmd(streamer_id)

            # cmd for cleaning up the streamer resources
            cleanup_cmd = self._get_cleanup_cmd(streamer)

            # cmd for submitting and executing ensemble streamers' eos cmds
            ensemble_eos_execute_cmd = self._get_eos_execute_cmd()

            streamer.add_eos_command(update_status_cmd)
            streamer.add_eos_command(cleanup_cmd)
            streamer.add_eos_command(ensemble_eos_execute_cmd)

    def _get_update_status_cmd(self, streamer_id):
        """Returns a command that updates the status of a streamer."""
        update_status_cmd = UpdateStreamerStatusCommand(
            streamer_id=streamer_id,
            new_status=StreamerStatus.COMPLETED,
            tracker=self._streamer_statuses
        )
        return ConcurrentCommand(
            command=update_status_cmd,
            executor=self._get_executor()
        )


    def _get_cleanup_cmd(self, streamer):
        """Returns a command that cleans up streamer resources."""
        return ConcurrentCommand(
            command=CleanupStreamerCommand(streamer),
            executor=self._get_executor()
        )

    def _get_eos_execute_cmd(self):
        """Returns a command that execute the ensemble streamer's eos commands if all streamers have finished."""
        queue_cmd = QueueCommand(self._eos_commands)

        # only submit and execute eos cmds if all streamers are finished
        condition = lambda: all(
            s in {StreamerStatus.COMPLETED, StreamerStatus.FAILED}
            for s in self._streamer_statuses.values()
        )

        cond_eos_cmd = ConditionalCommand(
            command=queue_cmd,
            condition=condition
        )

        return ConcurrentCommand(
            command=cond_eos_cmd,
            executor=self._get_executor()
        )


    def stream(self) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Ensemble streamer already running")

            self._running = True
            self._run_executor()

            streamer_ids = self.get_streamer_ids()
            for streamer_id in streamer_ids:
                self._streamers.get(streamer_id).stream()

    def streaming(self) -> bool:
        with self._lock:
            return self._running

    def stop(self) -> None:
        with self._lock:
            streamer_ids = self.get_streamer_ids()
            for streamer_id in streamer_ids:
                self._streamers.get(streamer_id).stop()

            self._stop_executor()
            self._running = False

    def wait_for_completion(self) -> None:
        with self._lock:
            streamer_ids = self.get_streamer_ids()
            for streamer_id in streamer_ids:
                self._streamers.get(streamer_id).wait_for_completion()
            self._stop_executor()
            self._running = False

    def _all_streamers_done(self) -> bool:
        """Returns True if all streamers have either COMPLETED or FAILED."""
        return all(v in {StreamerStatus.COMPLETED, StreamerStatus.FAILED} for v in self._streamer_statuses.values())

    def add_eos_command(self, command: Command) -> None:
        self._eos_commands.put(command)

    @staticmethod
    def _generate_streamer_id() -> str:
        """
        Generates a unique streamer identifier.
        """
        return str(uuid.uuid4())
