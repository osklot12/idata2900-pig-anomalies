from socket import socket
from typing import TypeVar

from src.data.structures.atomic_bool import AtomicBool
from src.network.messages.readers.stream_message_reader import StreamMessageReader
from src.network.messages.readers.stream_read_error import StreamReadError
from src.network.messages.requests.handlers.registry.request_handler_registry import RequestHandlerRegistry
from src.network.messages.requests.request import Request
from src.network.messages.serialization.message_deserializer import MessageDeserializer
from src.network.messages.serialization.message_serializer import MessageSerializer
from src.network.messages.writers.stream_message_writer import StreamMessageWriter
from src.network.network_config import NETWORK_MSG_LEN_FORMAT
from src.network.server.session.session import Session

T = TypeVar("T")


class ClientHandler:
    """Handles incoming client requests."""

    def __init__(self, client_socket: socket, serializer: MessageSerializer,
                 deserializer: MessageDeserializer[Request], handler_registry: RequestHandlerRegistry,
                 session: Session[T]):
        """
        Initializes a ClientHandler instance.

        Args:
            client_socket (socket.socket): the client socket
            serializer (MessageSerializer): the message serializer
            deserializer (MessageDeserializer): the message deserializer
            handler_registry (RequestHandlerRegistry): the request handler registry
            session (Session[T]): the network session
        """
        self._socket = client_socket
        self._msg_reader = StreamMessageReader(self._socket.makefile('rb'), NETWORK_MSG_LEN_FORMAT)
        self._msg_writer = StreamMessageWriter(self._socket.makefile('wb'), NETWORK_MSG_LEN_FORMAT)

        self._serializer = serializer
        self._deserializer = deserializer
        self._handler_registry = handler_registry
        self._session = session

        self._running = AtomicBool(False)

    def handle(self) -> None:
        """Handles the client request."""
        self._running = AtomicBool(True)
        recv_raw_msg = self._read_next_msg()
        while recv_raw_msg and self._running:
            recv_msg = self._deserializer.deserialize(recv_raw_msg)
            handler = self._handler_registry.get_handler(type(recv_msg))
            if not handler:
                raise RuntimeError(f"No handler registered for {recv_msg}")

            response = handler.handle(recv_msg)
            self._msg_writer.write(self._serializer.serialize(response))

            recv_raw_msg = self._read_next_msg()

        self._session.cleanup()

    def _read_next_msg(self) -> bytes:
        """Reads the next message in bytes."""
        msg = None

        try:
            msg = self._msg_reader.read()

        except (StreamReadError, OSError, ConnectionResetError, BrokenPipeError, ValueError) as e:
            print(f"[ClientHandler] Stream read error (disconnection or broken pipe): {e}")

        except Exception as e:
            print(f"[ClientHandler] Unexpected read error: {e}")

        return msg

    def stop(self) -> None:
        """Stops the client handler."""
        self._running.set(False)