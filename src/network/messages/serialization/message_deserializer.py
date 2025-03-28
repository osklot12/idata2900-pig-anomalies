from abc import ABC, abstractmethod
from typing import Generic

from typing_extensions import TypeVar

T = TypeVar('T', bound="Message")

class MessageDeserializer(ABC, Generic[T]):
    """An interface for message deserialization."""

    @abstractmethod
    def deserialize(self, message: bytes) -> T:
        """
        Deserializes a message from bytes.

        Args:
            message (bytes): the serialized message

        Returns:
            T: the deserialized message
        """
        raise NotImplementedError