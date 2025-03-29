from abc import ABC, abstractmethod
from typing import Type

from src.network.messages.requests.handlers.request_handler import RequestHandler
from src.network.messages.requests.request import Request


class RequestHandlerRegistry(ABC):
    """Interface for request handler registries."""

    @abstractmethod
    def register(self, request_type: Type[Request], handler: RequestHandler) -> None:
        """
        Registers a request handler.

        Args:
            request_type (Type[Request]): the request type associated with the RequestHandler
            handler (RequestHandler): the request handler
        """
        raise NotImplementedError

    @abstractmethod
    def get_handler(self, request_type: Type[Request]) -> RequestHandler:
        """
        Returns the RequestHandler for the given request type.

        Args:
            request_type (Type[Request]): the request type associated with the RequestHandler

        Returns:
            RequestHandler: the RequestHandler for the given request type
        """
        raise NotImplementedError
