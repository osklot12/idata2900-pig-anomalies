from abc import ABC, abstractmethod

from src.network.server.server_context import ServerContext


class Response(ABC):
    """An interface for network responses."""

    @abstractmethod
    def execute(self, context: ServerContext):
        """
        Executes the response command.

        Args:
            context (ServerContext): the server context to operate on
        """
        raise NotImplementedError