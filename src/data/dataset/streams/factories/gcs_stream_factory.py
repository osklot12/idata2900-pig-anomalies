from typing import TypeVar, Generic, List, Optional

from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.splitters.factories.splitter_factory import SplitterFactory
from src.data.dataset.splitters.string_set_splitter import StringSetSplitter
from src.data.dataset.streams.factories.managed_stream_factory import ManagedStreamFactory
from src.data.dataset.streams.factories.stream_factory import StreamFactory
from src.data.preprocessing.preprocessor import Preprocessor
from src.utils.gcs_credentials import GCSCredentials

T = TypeVar("T")


class GCSStreamFactory(Generic[T], ManagedStreamFactory[T]):
    """Factory for creating managed Google Cloud Storage (GCS) streams."""

    def __init__(self, gcs_creds: GCSCredentials, stream_factory: StreamFactory[T], splitter_factory: SplitterFactory[str],
                 split: DatasetSplit, preprocessors: Optional[List[Preprocessor[T]]] = None):
        """
        Initializes a GCSStreamFactory instance.

        Args:
            gcs_creds (GCSCredentials): Google Cloud Storage credentials
            stream_factory (StreamFactory[T]): factory for creating streams
            splitter_factory (SplitterFactory[str]): factory for creating splitters of string sets
            split (DatasetSplit): dataset split to create stream for
            preprocessors (Optional[List[Preprocessor[T]]]): optional list of preprocessors to use, defaults to None
        """
        self._gcs_creds = gcs_creds
        self._stream_factory = stream_factory
        self._splitter_factory = splitter_factory
        self._split = split
        self._preprocessors = preprocessors