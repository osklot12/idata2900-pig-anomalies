import time

from src.data.pipeline.cheap_pipeline import CheapPipeline
from src.network.messages.requests.get_frame_batch_request import GetFrameBatchRequest
from src.network.messages.requests.handlers.get_frame_batch_handler import GetFrameBatchHandler
from src.network.messages.requests.handlers.registry.simple_request_handler_registry import SimpleRequestHandlerRegistry
from src.network.messages.serialization.factories.pickle_deserializer_factory import PickleDeserializerFactory
from src.network.messages.serialization.factories.pickle_serializer_factory import PickleSerializerFactory
from src.network.server.network_server import NetworkServer


def run():
    pipeline = CheapPipeline()

    serializer_factory = PickleSerializerFactory()
    deserializer_factory = PickleDeserializerFactory()

    handler_registry = SimpleRequestHandlerRegistry()
    handler_registry.register(GetFrameBatchRequest, GetFrameBatchHandler(pipeline))

    server = NetworkServer(serializer_factory, deserializer_factory, handler_registry)
    server.run()

    try:
        pipeline.run()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()
        server.stop()


if __name__ == "__main__":
    run()
