from src.data.streamers.streamer import Streamer


class DummyFailingStreamer(Streamer):
    """A dummy streamer that fails (raises exception when streaming)."""

    def _stream(self) -> None:
        raise RuntimeError("Failed on purpose")