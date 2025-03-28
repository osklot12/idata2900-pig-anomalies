import pickle

from src.network.messages.message import Message
from src.network.messages.serialization.message_serializer import MessageSerializer


class PickleMessageSerializer(MessageSerializer):
    """A message serializer using Pickle."""

    def serialize(self, message: Message) -> bytes:
        return pickle.dumps(message)

