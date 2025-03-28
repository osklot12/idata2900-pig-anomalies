from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.client.client_context import ClientContext
from src.network.messages.message import Message

C = TypeVar('C')
R = TypeVar('R')

class Response(Message, Generic[C, R]):
    """An interface for network responses."""

    @abstractmethod
    def execute(self, context: C) -> R:
        """
        Executes the response command.

        Args:
            context (C): A context object.

        Returns:
            R: The response message.
        """
        raise NotImplementedError