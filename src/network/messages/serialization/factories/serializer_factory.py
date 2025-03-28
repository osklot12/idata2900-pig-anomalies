from abc import ABC, abstractmethod

from src.network.messages.serialization.message_serializer import MessageSerializer


class SerializerFactory(ABC):
    """An interface for serializer factories."""

    @abstractmethod
    def create_serializer(self) -> MessageSerializer:
        """
        Creates a MessageSerializer instance.

        Returns:
            MessageSerializer: the message serializer
        """
        raise NotImplementedError