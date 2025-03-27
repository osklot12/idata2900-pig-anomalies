from abc import ABC, abstractmethod

from src.network.client.client_context import ClientContext


class Response(ABC):
    """An interface for network responses."""

    @abstractmethod
    def execute(self, context: ClientContext):
        """
        Executes the response command.

        Args:
            context (ClientContext): the server context to operate on
        """
        raise NotImplementedError