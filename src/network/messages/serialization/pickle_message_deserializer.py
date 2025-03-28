import pickle

from src.network.messages.serialization.message_deserializer import MessageDeserializer, T

class PickleMessageDeserializer(MessageDeserializer[T]):
    """A message deserializer using Pickle."""

    def deserialize(self, message: bytes) -> T:
        return pickle.loads(message)