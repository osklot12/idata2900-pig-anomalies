from src.data.dataset.dataset_split import DatasetSplit
from src.data.dataset.streams.prefetcher import Prefetcher
from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_frame_instance_provider import NetworkFrameInstanceProvider


def run():
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    instance_provider = NetworkFrameInstanceProvider(client)
    prefetch = Prefetcher(instance_provider, DatasetSplit.TRAIN, 8, 20, 60)

    client.connect("10.0.0.1")

    prefetch.run()

    while True:
        print(prefetch.get_next_prefetched())




if __name__ == "__main__":
    run()