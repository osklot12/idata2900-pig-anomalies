from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.client.client_context import ClientContext
from src.network.messages.message import Message

T = TypeVar('T')

class Response(Message, Generic[T]):
    """An interface for network responses."""

    @abstractmethod
    def execute(self, context: ClientContext) -> T:
        """
        Executes the response command.

        Args:
            context (ClientContext): the server context to operate on
        """
        raise NotImplementedError