from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response

# any request
R = TypeVar("R", bound=Request)


class RequestHandler(Generic[R], ABC):
    """An interface for request extractors."""

    @abstractmethod
    def handle(self, request: R) -> Response:
        """
        Handles a request.

        Args:
            request (Request): the request to handle

        Returns:
            Response: the response to the request
        """
        raise NotImplementedError
