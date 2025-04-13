import socket

from src.network.client.network_client import NetworkClient
from src.network.messages.readers.stream_message_reader import StreamMessageReader
from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response
from src.network.messages.serialization.message_deserializer import MessageDeserializer
from src.network.messages.serialization.message_serializer import MessageSerializer
from src.network.messages.writers.stream_message_writer import StreamMessageWriter
from src.network.network_config import NETWORK_SERVER_PORT, NETWORK_MSG_LEN_FORMAT


class SimpleNetworkClient(NetworkClient):
    """A network client for requesting data from a server."""

    def __init__(self, serializer: MessageSerializer, deserializer: MessageDeserializer):
        """
        Initializes a NetworkClient instance.

        Args:
            serializer (MessageSerializer): the message serializer
            deserializer (MessageDeserializer): the message deserializer
        """
        self._serializer = serializer
        self._deserializer = deserializer
        self._sock = None
        self._reader = None
        self._writer = None

    def connect(self, server_ip: str) -> None:
        try:
            self._sock = socket.create_connection((server_ip, NETWORK_SERVER_PORT))
            self._reader = StreamMessageReader(self._sock.makefile("rb"), NETWORK_MSG_LEN_FORMAT)
            self._writer = StreamMessageWriter(self._sock.makefile("wb"), NETWORK_MSG_LEN_FORMAT)
            print(f"[SimpleNetworkClient] Connected to {server_ip}")
        except socket.error as e:
            raise ConnectionError(f"Failed to connect to {server_ip}: {e}")

    def send_request(self, request: Request) -> Response:
        if not self._sock:
            raise RuntimeError("Client is not connected to a server")

        try:
            self._writer.write(self._serializer.serialize(request))
            print(f"[SimpleNetworkClient] Sent {request}")
            response = self._deserializer.deserialize(self._reader.read())
            print(f"[SimpleNetworkClient] Received {response}")
        except socket.error as e:
            raise ConnectionError(f"Failed to send request to {request}: {e}")

        return response

    def disconnect(self) -> None:
        if self._sock:
            self._sock.close()
            self._sock = None
            self._reader = None
            self._writer = None