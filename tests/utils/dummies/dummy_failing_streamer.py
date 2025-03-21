from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer


class DummyFailingStreamer(ThreadedStreamer):
    """A dummy streamer that fails (raises exception when streaming)."""

    def _stream(self) -> None:
        raise RuntimeError("Failed on purpose")