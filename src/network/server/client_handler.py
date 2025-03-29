from socket import socket

from src.network.messages.readers.stream_message_reader import StreamMessageReader
from src.network.messages.requests.handlers.registry.simple_request_handler_registry import SimpleRequestHandlerRegistry
from src.network.messages.requests.request import Request
from src.network.messages.serialization.message_deserializer import MessageDeserializer
from src.network.messages.serialization.message_serializer import MessageSerializer
from src.network.messages.writers.stream_message_writer import StreamMessageWriter
from src.network.network_config import NETWORK_MSG_LEN_FORMAT


class ClientHandler:
    """Handles incoming client requests."""

    def __init__(self, client_socket: socket,
                 serializer: MessageSerializer,
                 deserializer: MessageDeserializer[Request],
                 handler_registry: SimpleRequestHandlerRegistry):
        """
        Initializes a ClientHandler instance.

        Args:
            client_socket (socket.socket): the client socket
            serializer (MessageSerializer): the message serializer
            deserializer (MessageDeserializer): the message deserializer
            handler_registry (SimpleRequestHandlerRegistry): the request handler registry
        """
        self._socket = client_socket
        self._msg_reader = StreamMessageReader(self._socket.makefile('rb'), NETWORK_MSG_LEN_FORMAT)
        self._msg_writer = StreamMessageWriter(self._socket.makefile('wb'), NETWORK_MSG_LEN_FORMAT)

        self._serializer = serializer
        self._deserializer = deserializer

        self._handler_registry = handler_registry

    def handle(self) -> None:
        """Handles the client request."""
        recv_raw_msg = self._msg_reader.read()
        while recv_raw_msg:
            recv_msg = self._deserializer.deserialize(recv_raw_msg)
            handler = self._handler_registry.get_handler(recv_msg)
            if not handler:
                raise RuntimeError(f"No handler registered for {recv_msg}")

            response = handler.handle(recv_msg)
            self._msg_writer.write(self._serializer.serialize(response))

            recv_raw_msg = self._msg_reader.read()