from abc import ABC, abstractmethod

from src.network.messages.response.response import Response


class Client(ABC):

    @abstractmethod
    def receive(self) -> Response:
        """
        Recieves a response from the server
        """
        raise NotImplementedError
