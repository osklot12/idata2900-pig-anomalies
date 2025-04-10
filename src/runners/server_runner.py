import time

from src.network.messages.serialization.factories.pickle_deserializer_factory import PickleDeserializerFactory
from src.network.messages.serialization.factories.pickle_serializer_factory import PickleSerializerFactory
from src.network.server.network_server import NetworkServer


def main():
    server = NetworkServer(PickleSerializerFactory(), PickleDeserializerFactory())
    server.run()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()