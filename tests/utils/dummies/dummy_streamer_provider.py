from src.data.streaming.streamers.providers.streamer_factory import StreamerFactory
from src.data.streaming.streamers.linear_streamer import LinearStreamer
from tests.utils.dummies.dummy_streamer import DummyStreamer


class DummyStreamerProvider(StreamerFactory):
    """A dummy streamer provider for testing."""

    def __init__(self, n_streamers: int = -1, streamer_wait_time: float = .5):
        """
        Initializes a DummyStreamerProvider instance.

        Args:
            n_streamers (int): Number of streamers to provide, -1 for infinite streamers.
            streamer_wait_time (int): Number of seconds to simulate streaming in each streamer.
        """
        if n_streamers < -1:
            raise ValueError(f"n_streamers must be -1 or greater")

        self._n_streamers = n_streamers
        self._streamer_wait_time = streamer_wait_time

        self._counter = 0

    def create_streamer(self) -> LinearStreamer:
        result = None

        streamers_left = self._n_streamers - self._counter
        if not streamers_left == 0:
            result = DummyStreamer(self._streamer_wait_time)
            self._counter += 1

        return result
