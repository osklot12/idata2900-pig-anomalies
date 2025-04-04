from src.data.dataset.dataset_split import DatasetSplit
from src.data.streaming.prefetchers.batch_prefetcher import BatchPrefetcher
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider
import time


def main():
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect("10.0.0.1")

    provider = NetworkFrameInstanceProvider(client)

    prefetcher = BatchPrefetcher(provider, DatasetSplit.TRAIN, batch_size=8)
    prefetcher.run()

    while True:
        time.sleep(.1)
        batch = prefetcher.get()
        print(f"Got batch: {batch}")

if __name__ == "__main__":
    main()