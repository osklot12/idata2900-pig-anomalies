import time

from src.data.streamers.streamer import Streamer
from src.data.streamers.streamer_status import StreamerStatus


class DummyStreamer(Streamer):
    """A dummy streamer for testing."""

    def __init__(self, wait_time: float = 0.1):
        super().__init__()
        self.wait_time = wait_time

    def _stream(self) -> None:
        time.sleep(self.wait_time)

        if self._is_requested_to_stop():
            # simulating preemptive stop
            self._set_status(StreamerStatus.STOPPED)

        else:
            # simulating completion
            self._set_status(StreamerStatus.COMPLETED)