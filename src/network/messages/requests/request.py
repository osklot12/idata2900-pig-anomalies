from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.messages.message import Message
from src.network.messages.response.response import Response

# server-side context
S = TypeVar('S')

# return-type of response
R = TypeVar('R')

class Request(Message, Generic[S, R]):
    """
    An interface for requests.

    Type Parameters:
        S (ServerContext): context used on the server side to execute the request
        R (ReturnType): the final result produced when the response is executed
    """

    @abstractmethod
    def execute(self, context: S) -> Response[R]:
        """
        Executes the request.

        Args:
            context (C): A context object.

        Returns:
            Response: A response message.
        """
        raise NotImplementedError