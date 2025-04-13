from src.data.dataset.streams.managed.factories.split_stream_factory import SplitStreamFactory
from src.network.messages.requests.close_stream_request import CloseStreamRequest
from src.network.messages.requests.get_batch_request import GetBatchRequest
from src.network.messages.requests.handlers.close_stream_handler import CloseStreamHandler
from src.network.messages.requests.handlers.get_batch_handler import GetBatchHandler
from src.network.messages.requests.handlers.open_stream_handler import OpenStreamHandler
from src.network.messages.requests.handlers.registry.factories.handler_registry_factory import HandlerRegistryFactory, T
from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.messages.requests.handlers.registry.simple_request_handler_registry import SimpleRequestHandlerRegistry
from src.network.messages.requests.open_stream_request import OpenStreamRequest
from src.network.server.session.session import Session


class DefaultHandlerRegistryFactory(HandlerRegistryFactory):
    """Factory for creating request handler registries with all the default handlers."""

    def __init__(self, stream_factory: SplitStreamFactory):
        """
        Initializes a DefaultHandlerRegistryFactory instance.

        Args:
            stream_factory (SplitStreamFactory): factory for creating dataset split streams
        """
        self._stream_factory = stream_factory

    def create_registry(self, session: Session[T]) -> RequestHandlerRegistry:
        registry = SimpleRequestHandlerRegistry()

        registry.register(OpenStreamRequest, OpenStreamHandler(session=session, stream_factory=self._stream_factory))
        registry.register(GetBatchRequest, GetBatchHandler(session=session))
        registry.register(CloseStreamRequest, CloseStreamHandler(session=session))

        return registry