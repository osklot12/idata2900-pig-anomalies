from src.network.messages.requests.close_stream_request import CloseStreamRequest
from src.network.messages.requests.handlers.close_stream_handler import CloseStreamHandler
from src.network.messages.requests.handlers.dataset_stream_factories import DatasetStreamFactories
from src.network.messages.requests.handlers.open_stream_handler import OpenStreamHandler
from src.network.messages.requests.handlers.read_stream_handler import ReadStreamHandler
from src.network.messages.requests.handlers.registry.factories.handler_registry_factory import HandlerRegistryFactory, T
from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.messages.requests.handlers.registry.simple_request_handler_registry import SimpleRequestHandlerRegistry
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.messages.requests.read_stream_request import ReadStreamRequest
from src.network.server.session.session import Session


class DefaultHandlerRegistryFactory(HandlerRegistryFactory):
    """Factory for creating request handler registries with all the default handlers."""

    def __init__(self, stream_factories: DatasetStreamFactories):
        """
        Initializes a DefaultHandlerRegistryFactory instance.

        Args:
            stream_factories (DatasetStreamFactories): factories for creating dataset streams
        """
        self._stream_factories = stream_factories

    def create_registry(self, session: Session[T]) -> RequestHandlerRegistry:
        registry = SimpleRequestHandlerRegistry()

        registry.register(OpenStreamRequest, OpenStreamHandler(session=session, stream_factories=self._stream_factories))
        registry.register(ReadStreamRequest, ReadStreamHandler(session=session))
        registry.register(CloseStreamRequest, CloseStreamHandler(session=session))

        return registry