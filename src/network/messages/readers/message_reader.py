from abc import ABC, abstractmethod

class MessageReader(ABC):
    """An interface for message readers."""

    @abstractmethod
    def read(self) -> bytes:
        """
        Reads a single complete message from the socket.

        Returns:
            bytes: the raw payload
        """
        raise NotImplementedError