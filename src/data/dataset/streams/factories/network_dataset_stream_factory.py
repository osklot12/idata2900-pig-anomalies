from typing import TypeVar, Generic

from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.closable_stream import ClosableStream
from src.data.dataset.streams.factories.stream_factory import ClosableStreamFactory
from src.data.dataset.streams.network_stream import NetworkStream
from src.data.dataset.streams.pipeline_stream import PipelineStream
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.pipeline_builder import PipelineBuilder
from src.data.pipeline.preprocessor import Preprocessor
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer

# stream data type
T = TypeVar("T")

# sink input data type
B = TypeVar("B")

class NetworkDatasetStreamFactory(Generic[T], ClosableStreamFactory[T]):
    """Factory for creating network dataset streams."""

    def __init__(self, server_ip: str, split: DatasetSplit,
                 pipeline: PipelineBuilder[CompressedAnnotatedFrame, B] = Pipeline(Preprocessor(ZlibDecompressor()))):
        """
        Initializes a NetworkDatasetStreamFactory instance.

        Args:
            server_ip (str): the server ip address
            split (DatasetSplit): the dataset split to get stream for
            pipeline (PipelineBuilder[T]): the data processing pipeline
        """
        self._server_ip = server_ip
        self._split = split
        self._pipeline = pipeline

    def create_stream(self) -> ClosableStream[T]:
        client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
        client.connect(self._server_ip)

        network_stream = NetworkStream(client=client, split=self._split, data_type=CompressedAnnotatedFrame)
        prefetcher = Prefetcher(network_stream)
        stream = PipelineStream(source=prefetcher, pipeline=self._pipeline)

        prefetcher.run()
        return stream