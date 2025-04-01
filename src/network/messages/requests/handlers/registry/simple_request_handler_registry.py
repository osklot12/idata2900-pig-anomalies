from typing import Dict, Type

from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.request import Request


class SimpleRequestHandlerRegistry(RequestHandlerRegistry):
    """A registry for request handlers."""

    def __init__(self):
        """Initializes a RequestHandlerRegistry."""
        self._handlers: Dict[Type[Request], RequestHandler] = {}

    def register(self, request_type: Type[Request], handler: RequestHandler) -> None:
        self._handlers[request_type] = handler

    def get_handler(self, request: Request) -> RequestHandler:
        return self._handlers.get(type(request))
