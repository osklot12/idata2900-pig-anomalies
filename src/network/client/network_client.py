from abc import ABC, abstractmethod

from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response

class NetworkClient(ABC):
    """A network client for sending requests to servers."""

    @abstractmethod
    def connect(self, server_ip: str) -> None:
        """
        Connects to the server.

        Args:
            server_ip (str): the IP address of the network server
        """
        raise NotImplementedError

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

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnects from the server."""
        raise NotImplementedError
