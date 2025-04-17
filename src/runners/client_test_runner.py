import time

from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.network_stream import NetworkStream
from src.data.dataset.streams.pipeline_stream import PipelineStream
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.preprocessor import Preprocessor
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer

SERVER_IP = "10.0.0.1"


def main():
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect(SERVER_IP)

    network_stream = NetworkStream(client=client, split=DatasetSplit.TRAIN, data_type=CompressedAnnotatedFrame)
    prefetcher = Prefetcher(network_stream)
    pipeline = Pipeline(Preprocessor(ZlibDecompressor()))
    stream = PipelineStream(source=prefetcher, pipeline=pipeline)

    prefetcher.run()
    instance = stream.read()
    try:
        while instance:
            print(f"[Test] Read instance {instance}")
            time.sleep(1)

    except KeyboardInterrupt:
        print("[Test] Stopping...")
        prefetcher.stop()