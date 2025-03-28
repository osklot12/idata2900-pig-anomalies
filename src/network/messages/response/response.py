from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.messages.message import Message

# return type of response
R = TypeVar('R')

class Response(Message, Generic[R]):
    """
    An interface for network responses.

    Type Parameters:
        R (ReturnType): the final result produced when the response is executed
    """

    @abstractmethod
    def execute(self) -> R:
        """
        Executes the response command.

        Returns:
            R: the response result.
        """
        raise NotImplementedError