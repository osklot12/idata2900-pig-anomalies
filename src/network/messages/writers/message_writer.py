from abc import ABC, abstractmethod

class MessageWriter(ABC):
    """An interface for message writers."""

    @abstractmethod
    def write(self, message: bytes) -> None:
        """
        Writes a message.

        Args:
            message (bytes): the message to write
        """
        raise NotImplementedError