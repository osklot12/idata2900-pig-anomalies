from src.data.dataset.streams.factories.dataset_stream_factory import DatasetStreamFactory, T
from src.data.dataset.streams.stream import Stream


class NorsvinStreamFactory(DatasetStreamFactory):
    """Factory for creating training set streams for the Norsvin dataset."""

    def __init__(self, service_account_path: str):
        """
        Initializes a NorsvinStreamFactory instance.

        Args:
            service_account_path: path to the service account json file
        """
        self._service_account_path = service_account_path

    def create_train_stream(self) -> Stream[T]:
        pass

    def create_validation_stream(self) -> Stream[T]:
        pass

    def create_test_stream(self) -> Stream[T]:
        pass