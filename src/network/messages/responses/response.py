from abc import abstractmethod
from typing import TypeVar, Generic

from src.network.messages.message import Message

# return type of responses
R = TypeVar('R')

class Response(Message, Generic[R]):
    """
    An interface for network responses.

    Type Parameters:
        R (ReturnType): the final result produced when the responses is executed
    """

    @abstractmethod
    def execute(self) -> R:
        """
        Executes the responses command.

        Returns:
            R: the responses result.
        """
        raise NotImplementedError