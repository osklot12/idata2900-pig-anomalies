from abc import ABC, abstractmethod
from src.network.messages.response.response import Response
from src.network.server.server_context import ServerContext


class Request(ABC):
    """An interface for requests."""

    @abstractmethod
    def execute(self, context: ServerContext) -> Response:
        """
        Executes the request.

        Args:
            context (ServerContext): The context of the request to operate on.
        """
        raise NotImplementedError