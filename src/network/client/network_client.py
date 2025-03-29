from abc import ABC, abstractmethod

from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response

class NetworkClient(ABC):
    """A network client for sending requests to servers."""

    @abstractmethod
    def send_request(self, request: Request) -> Response:
        """
        Sends a request to the server.

        Args:
            request (Request): the request to send

        Returns:
            Response: the server response
        """
        raise NotImplementedError