from src.data.streaming.streamers.concurrent_streamer import ConcurrentStreamer


class DummyFailingStreamer(ConcurrentStreamer):
    """A dummy streamer that fails (raises exception when streaming)."""

    def _stream(self) -> None:
        raise RuntimeError("Failed on purpose")