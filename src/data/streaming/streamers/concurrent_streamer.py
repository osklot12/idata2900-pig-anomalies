import queue
import threading
from abc import abstractmethod

from src.command.command import Command
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.streamer_status import StreamerStatus
from src.data.structures.atomic_bool import AtomicBool


class ConcurrentStreamer(Streamer):
    """A threaded streamer, providing base functionality for streaming asynchronously."""

    def __init__(self):
        """Initializes a Streamer instance."""
        self._eos_commands = queue.SimpleQueue()

        self._thread = None
        self._stream_lock = threading.Lock()
        self._release = AtomicBool(True)

        self._requested_stop = False
        self._stop_lock = threading.Lock()

        self._status = StreamerStatus.PENDING
        self._status_lock = threading.Lock()

    def start_streaming(self) -> None:
        with self._stream_lock:
            if self._thread:
                raise RuntimeError("Streamer is already running")

            self._release.set(False)
            self._set_status(StreamerStatus.STREAMING)
            self._thread = threading.Thread(target=self._stream_worker)
            self._thread.start()

    def _stream_worker(self) -> None:
        """Stream worker for streaming in the background."""
        try:
            self._setup_stream()
            self._set_status(self._stream())

        except Exception as e:
            self._set_status(StreamerStatus.FAILED)

        finally:
            while not self._eos_commands.empty():
                cmd = self._eos_commands.get()
                cmd.execute()

    @abstractmethod
    def _stream(self) -> StreamerStatus:
        """
        Implementation specific streaming logic running in the background.

        Returns:
            StreamerStatus: The status of the streamer.
        """
        raise NotImplementedError()

    def _setup_stream(self) -> None:
        """Can be overridden for custom setup when start_streaming() is called."""
        pass

    def stop_streaming(self) -> None:
        with self._stream_lock:
            self._release.set(True)
            self._request_stop()
            self._stop()
            self._safe_join()
            self._thread = None

    def _stop(self) -> None:
        """Default stop behavior (can be overridden)."""
        pass

    def wait_for_completion(self) -> None:
        self._safe_join()
        self._wait_for_completion()

    def _wait_for_completion(self) -> None:
        """Implementation specific logic ran after joining the worker thread."""
        pass

    def get_status(self) -> StreamerStatus:
        """
        Returns the current status of the streamer.

        Returns:
            StreamerStatus: The status of the streamer.
        """
        with self._status_lock:
            return self._status

    def _set_status(self, status: StreamerStatus) -> None:
        """Sets the status of the streamer thread-safe."""
        with self._status_lock:
            self._status = status

    def add_eos_command(self, command: Command) -> None:
        """
        Adds a command that executes on end of streams.

        Args:
            command (Command): The command to execute.
        """
        self._eos_commands.put(command)

    def _safe_join(self) -> None:
        """Safely joins the worker thread."""
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def _request_stop(self) -> None:
        """Requests to stop the streamer."""
        with self._stop_lock:
            self._requested_stop = True

    def _is_requested_to_stop(self) -> bool:
        """Indicates whether the streamer has been requested to stop."""
        with self._stop_lock:
            return self._requested_stop

    def get_releaser(self) -> AtomicBool:
        """
        Returns a releaser that can be used to release resources.

        Returns:
            AtomicBool: the releaser
        """
        return self._release