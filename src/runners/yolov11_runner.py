from src.network.client.simple_network_client import SimpleNetworkClient
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer
from src.network.network_dataset_stream import NetworkDatasetStream


def main():
    # connect client
    client = SimpleNetworkClient(PickleMessageSerializer(), PickleMessageDeserializer())
    client.connect("10.0.0.1")

    train_stream = NetworkDatasetStream(client, )

if __name__ == "__main__":
    main()