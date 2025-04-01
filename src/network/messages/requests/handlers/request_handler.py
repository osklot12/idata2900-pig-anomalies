from abc import ABC, abstractmethod

from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response

class RequestHandler(ABC):
    """An interface for request extractors."""

    @abstractmethod
    def handle(self, request: Request) -> Response:
        """
        Handles a request.

        Args:
            request (Request): the request to handle

        Returns:
            Response: the response to the request
        """
        raise NotImplementedError