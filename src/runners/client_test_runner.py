import time

from src.data.dataclasses.compressed_annotated_frame import CompressedAnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.factories.network_dataset_stream_factory import NetworkDatasetStreamFactory
from src.data.dataset.streams.network_stream import NetworkStream
from src.data.dataset.streams.pipeline_stream import PipelineStream
from src.data.dataset.streams.prefetcher import Prefetcher
from src.data.pipeline.pipeline import Pipeline
from src.data.pipeline.preprocessor import Preprocessor
from src.data.processing.zlib_decompressor import ZlibDecompressor
from src.models.twod.yolo.x.yolox_dataset import YOLOXDataset
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer

SERVER_IP = "10.0.0.1"


def run_train_stream():
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

            instance = stream.read()

    except KeyboardInterrupt:
        print("[Test] Stopping...")


def run_val_stream():
    factory = NetworkDatasetStreamFactory(server_ip=SERVER_IP, split=DatasetSplit.VAL)
    dataset = YOLOXDataset(stream_factory=factory, batch_size=8, n_batches=430)

    try:
        while True:
            for batch in dataset:
                print(f"[Test] Read batch of size {len(batch[0])}")

    except KeyboardInterrupt:
        print("[Test] Stopping...")



if __name__ == "__main__":
    run_val_stream()
