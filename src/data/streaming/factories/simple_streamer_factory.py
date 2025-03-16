from src.data.dataset.providers.dataset_file_pair_provider import DatasetFilePairProvider
from src.data.loading.factories.loader_factory import LoaderFactory
from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.streamer import Streamer


class FileStreamerFactory(StreamerFactory):
    """A simple streamer factory."""

    def __init__(self, loader_factory: LoaderFactory, file_provider: DatasetFilePairProvider):
        """
        Initializes a FileStreamerFactory instance.

        Args:
            loader_factory: the loader factory
            file_provider: the dataset file provider
        """
        self._loader_factory = loader_factory
        self._file_provider = file_provider

    def get_next_streamer(self) -> Streamer:
        file_pair = self._file_provider.get_file_pair()
        frame_