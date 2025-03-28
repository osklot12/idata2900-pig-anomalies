from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.messages.message import Message
from src.network.messages.response.response import Response
from src.network.server.server_context import ServerContext

C = TypeVar('C')
R = TypeVar('R')

class Request(Message, Generic[C, R]):
    """An interface for requests."""

    @abstractmethod
    def execute(self, context: C) -> Response[?, R]:
        """
        Executes the request.

        Args:
            context (C): A context object.

        Returns:
            Response: A response message.
        """
        raise NotImplementedError