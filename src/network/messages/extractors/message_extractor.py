from abc import ABC, abstractmethod
from socket import socket

class MessageExtractor(ABC):
    """An interface for message extractors."""

    @abstractmethod
    def extract(self, sock: socket) -> bytes:
        """
        Reads a single complete message from the socket.

        Returns:
            bytes: the raw payload
        """
        raise NotImplementedError