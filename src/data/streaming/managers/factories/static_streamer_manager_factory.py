from src.data.streaming.managers.factories.streamer_manager_factory import StreamerManagerFactory
from src.data.streaming.managers.static_streamer_manager import StaticStreamerManager
from src.data.streaming.managers.streamer_manager import StreamerManager


class StaticStreamerManagerFactory(StreamerManagerFactory):
    """Factory for creating StaticStreamerManager instances."""

    def create_manager(self) -> StreamerManager:
        return StaticStreamerManager()