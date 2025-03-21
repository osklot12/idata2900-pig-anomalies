import concurrent.futures
import threading
from typing import Type

from src.data.streaming.managers.runnable_streamer_manager import RunnableStreamerManager
from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer
from src.data.streaming.managers.streamer_manager import StreamerManager
from src.data.streaming.factories.streamer_factory import StreamerFactory


class DockingStreamerManager(RunnableStreamerManager, StreamerManager):
    """
    A stream manager effective for large sets of finite streams, maintaining a constant number of streams at all time.
    The Streamers "dock" the manager, before leaving and making space for the next streamer.
    """

    def __init__(self, streamer_provider: StreamerFactory, n_streamers: int):
        """
        Initializes a new instance of the DockingStreamManager class.

        Args:
            streamer_provider (Type[StreamerFactory]): Provides streamers.
            n_streamers (int): The number of streamers to maintain at all times.
        """
        if n_streamers < 1:
            raise ValueError("n_streamers must be greater than 0")

        super().__init__()

        self._streamer_provider = streamer_provider
        self._n_streamers = n_streamers

        self._running = False

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=self._n_streamers)
        self._lock = threading.Lock()

    def run(self) -> None:
        self._running = True
        for _ in range(self._n_streamers):
            self._start_new_streamer()

    def _start_new_streamer(self):
        """Fetches a new streamer, assigns it an ID, and submits it to the executor."""
        streamer = self._streamer_provider.get_next_streamer()

        # only start next streamer if the streamer exists
        if streamer:
            streamer_id = self._add_streamer(streamer)

            streamer.start_streaming()
            future = self._executor.submit(self._manage_streamer, streamer, streamer_id)
            future.add_done_callback(self._on_streamer_done)

    @staticmethod
    def _manage_streamer(streamer: ThreadedStreamer, streamer_id: str) -> str:
        """Manages the streamer."""
        streamer.wait_for_completion()
        streamer.stop()
        return streamer_id

    def _on_streamer_done(self, future: concurrent.futures.Future) -> None:
        """Callback for streamers done streaming."""
        with self._lock:
            if self._running:
                streamer_id = future.result()
                self._remove_streamer(streamer_id)
                self._start_new_streamer()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        for streamer_id in self.get_streamer_ids():
            streamer = self.get_streamer(streamer_id)
            if streamer:
                streamer.stop()
            self._remove_streamer(streamer_id)

        self._executor.shutdown(wait=True)

    def n_active_streamers(self) -> int:
        return len(self._get_streamers())