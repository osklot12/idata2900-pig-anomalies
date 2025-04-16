from typing import List

from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.network_stream import NetworkStream
from src.data.dataset.streams.prefetcher import Prefetcher
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer


def main():
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect("10.0.0.1")

    stream = NetworkStream(client=client, split=DatasetSplit.TRAIN, batch_type=AnnotatedFrame, batch_size=8)
    prefetcher = Prefetcher[List[AnnotatedFrame]](stream=stream)
    prefetcher.run()

    while True:
        batch = prefetcher.read()
        for frame in batch:
            print(f"Received frame {frame.index} from {frame.source.source_id}")


if __name__ == "__main__":
    main()