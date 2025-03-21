import time

from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class DummyStreamer(ThreadedStreamer):
    """A dummy streamer for testing."""

    def __init__(self, wait_time: float = 0.1):
        super().__init__()
        self.wait_time = wait_time

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.FAILED

        time.sleep(self.wait_time)

        if self._is_requested_to_stop():
            # simulating preemptive stop
            result = StreamerStatus.STOPPED

        else:
            # simulating completion
            result = StreamerStatus.COMPLETED

        return result