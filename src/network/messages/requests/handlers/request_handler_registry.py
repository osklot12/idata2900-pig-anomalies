from typing import Dict, Type

from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.request import Request


class RequestHandlerRegistry:
    """A registry for request handlers."""

    def __init__(self):
        """Initializes a RequestHandlerRegistry."""
        self._handlers: Dict[Type[Request], RequestHandler] = {}

    def register(self, request_type: Type[Request], handler: RequestHandler) -> None:
        """
        Registers a request handler.

        Args:
            request_type (Type[Request]): the request type to register
            handler (RequestHandler): the request handler to register
        """
        self._handlers[request_type] = handler

    def get_handler(self, request: Request) -> RequestHandler:
        """
        Returns the handler for the given request.

        Returns:
            RequestHandler: the handler for the given request, or None if no handler exists
        """
        return self._handlers.get(type(request))
