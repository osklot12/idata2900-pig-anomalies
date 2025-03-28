from abc import ABC, abstractmethod

from src.network.messages.message import Message


class MessageSerializer(ABC):
    """An interface for message serialization."""

    @abstractmethod
    def serialize(self, message: Message) -> bytes:
        """
        Serializes a message into bytes.

        Args:
            message (Message): the message to serialize

        Returns:
            bytes: the serialized message
        """
        raise NotImplementedError
