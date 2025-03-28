from src.network.messages.serialization.factories.serializer_factory import SerializerFactory
from src.network.messages.serialization.message_serializer import MessageSerializer
from src.network.messages.serialization.pickle_message_serializer import PickleMessageSerializer


class PickleSerializerFactory(SerializerFactory):
    """A factory for creating PickleMessageSerializer instances."""

    def create_serializer(self) -> MessageSerializer:
        return PickleMessageSerializer()