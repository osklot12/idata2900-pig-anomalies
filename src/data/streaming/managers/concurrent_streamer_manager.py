import concurrent.futures
import threading
from abc import abstractmethod
from functools import partial

from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.managers.streamer_registry import StreamerRegistry
from src.data.streaming.streamers.producer_streamer import ProducerStreamer
from src.data.structures.atomic_bool import AtomicBool


class ConcurrentStreamerManager(StreamerManager, StreamerRegistry):
    """A streamer manager that handles streamers asynchronously."""

    def __init__(self, max_streamers: int):
        """
        Initializes a ConcurrentStreamerManager instance.

        Args:
            max_streamers (int): the maximum number of concurrent streamers
        """
        super().__init__()

        self._max_streamers = max_streamers

        self._running = AtomicBool(False)

        self._executor = concurrent.futures.ThreadPoolExecutor(self._max_streamers)
        self._lock = threading.Lock()

    def run(self) -> None:
        print(f"[ConcurrentStreamerManager] Running...")
        with self._lock:
            if self._running:
                raise RuntimeError("StreamerManager already running")
            self._running.set(True)
            self._setup()

    @abstractmethod
    def _setup(self) -> None:
        """Sets up for running. Cannot stop the manager while setup is executing."""
        raise NotImplementedError

    def _launch_streamer(self, streamer: ProducerStreamer) -> None:
        """
        Launches a streamer.

        Args:
            streamer (Producer): the streamer to launch
        """

        if not self._running:
            raise RuntimeError("Cannot launch streamer when manager is not running.")

        if streamer is None:
            raise RuntimeError("Streamer cannot be None")

        streamer_id = self._add_streamer(streamer)
        streamer.start_streaming()
        print(f"[ConcurrentStreamerManager] Launched streamer...")
        future = self._executor.submit(self._run_streamer, streamer, streamer_id)
        future.add_done_callback(partial(self._on_streamer_done, streamer_id=streamer_id))

    @abstractmethod
    def _run_streamer(self, streamer: ProducerStreamer, streamer_id: str) -> None:
        """
        Manages the streamer after launched.

        Args:
            streamer (Producer): the streamer to manage
            streamer_id (str): the streamer id
        """
        raise NotImplementedError

    def _on_streamer_done(self, future: concurrent.futures.Future, streamer_id: str) -> None:
        """
        Callback called when a streamer is done.

        Args:
            streamer_id (str): the streamer id
            future (concurrent.futures.Future): the future returned from the background task running `_manage_streamer`
        """
        with self._lock:
            if self._running:
                try:
                    future.result()
                    self._handle_done_streamer(streamer_id)
                except Exception as e:
                    self._handle_crashed_streamer(streamer_id, e)

    @abstractmethod
    def _handle_done_streamer(self, streamer_id: str) -> None:
        """
        Handles a finished streamer.

        Args:
            streamer_id (str): the streamer id
        """
        raise NotImplementedError

    @abstractmethod
    def _handle_crashed_streamer(self, streamer_id: str, e: Exception) -> None:
        """
        Handles a crashed streamer.

        Args:
            streamer_id (str): the streamer id
            e (Exception): the exception raised
        """
        raise NotImplementedError

    def n_active_streamers(self) -> int:
        return len(self._get_streamers())

    def stop(self) -> None:
        with self._lock:
            self._running.set(False)
        self._stop()

        for streamer_id in self.get_streamer_ids():
            streamer = self.get_streamer(streamer_id)
            if streamer:
                streamer.stop_streaming()

        try:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
        finally:
            for streamer_id in self.get_streamer_ids():
                self._remove_streamer(streamer_id)

    def _stop(self) -> None:
        """Custom stop logic to run before shutting down executor and stopping and removing streamers."""
        pass