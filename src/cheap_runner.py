import time

from src.data.pipeline.cheap_pipeline import CheapPipeline
from src.network.messages.serialization.factories.pickle_deserializer_factory import PickleDeserializerFactory
from src.network.messages.serialization.factories.pickle_serializer_factory import PickleSerializerFactory
from src.network.server.network_server import NetworkServer


def run():
    pipeline = CheapPipeline()
    serializer_factory = PickleSerializerFactory()
    deserializer_factory = PickleDeserializerFactory()

    try:
        print("Pipeline is running.")
        pipeline.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping pipeline...")
        pipeline.stop()
        print("Pipeline stopped.")


if __name__ == "__main__":
    run()
