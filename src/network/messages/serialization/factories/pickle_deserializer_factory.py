from src.network.messages.serialization.factories.deserializer_factory import DeserializerFactory
from src.network.messages.serialization.message_deserializer import MessageDeserializer
from src.network.messages.serialization.pickle_message_deserializer import PickleMessageDeserializer


class PickleDeserializerFactory(DeserializerFactory):
    """A factory for creating PickleMessageDeserializer instances."""

    def create_deserializer(self) -> MessageDeserializer:
        return PickleMessageDeserializer()