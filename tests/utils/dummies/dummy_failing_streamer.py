from src.data.streaming.streamers import Streamer


class DummyFailingStreamer(Streamer):
    """A dummy streamer that fails (raises exception when streaming)."""

    def _stream(self) -> None:
        raise RuntimeError("Failed on purpose")