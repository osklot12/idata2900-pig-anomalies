from src.data.dataset.factories.dataset_entity_factory import DatasetEntityFactory
from src.data.dataset.providers.dataset_instance_provider import DatasetInstanceProvider
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.streaming.aggregator_manager import AggregatorManager
from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer


class FileStreamerFactory(StreamerFactory):
    """A simple streamer factory."""

    def __init__(self, file_provider: DatasetInstanceProvider, entity_factory: DatasetEntityFactory):
        """
        Initializes a FileStreamerFactory instance.

        Args:
            file_provider (DatasetInstanceProvider): a provider of video-annotations pairs
            entity_factory (DatasetEntityFactory): a factory that creates DatasetEntity objects from the provided files
        """
        self._file_provider = file_provider
        self._entity_factory = entity_factory

    def get_next_streamer(self) -> ThreadedStreamer:
