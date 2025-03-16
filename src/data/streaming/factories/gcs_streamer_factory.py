from src.data.streaming.factories.streamer_factory import StreamerFactory
from src.data.streaming.streamers.streamer import Streamer


class GCSStreamerFactory(StreamerFactory):
    """A factory that creates streamers streaming from Google Cloud Storage."""

    def __init__(self, bucket_name: str, service_account_path: str):
        """
        Initializes a GCSStreamerFactory instance.

        Args:
            bucket_name (str): the name of the bucket on Google Cloud Storage
            service_account_path (str): the path to the service account json file
        """
        self._bucket_name = bucket_name
        self._service_account_path = service_account_path


    def get_next_streamer(self) -> Streamer:
        pass