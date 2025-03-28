from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.messages.message import Message
from src.network.messages.response.response import Response
from src.network.server.server_context import ServerContext

T = TypeVar('T')

class Request(Message, Generic[T]):
    """An interface for requests."""

    @abstractmethod
    def execute(self, context: ServerContext) -> Response[T]:
        """
        Executes the request.

        Args:
            context (ServerContext): The context of the request to operate on.
        """
        raise NotImplementedError