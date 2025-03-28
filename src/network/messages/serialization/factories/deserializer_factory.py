from abc import ABC, abstractmethod

from src.network.messages.serialization.message_deserializer import MessageDeserializer


class DeserializerFactory(ABC):
    """An interface for deserializer factories."""

    @abstractmethod
    def create_deserializer(self) -> MessageDeserializer:
        """
        Creates a MessageDeserializer instance.

        Returns:
            MessageDeserializer: the deserializer instance
        """
        raise NotImplementedError