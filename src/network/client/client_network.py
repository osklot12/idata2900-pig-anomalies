import socket
import struct
from typing import TypeVar

from src.network.messages.readers.message_reader import MessageReader
from src.network.messages.readers.stream_message_reader import StreamMessageReader
from src.network.messages.requests.request import Request
from src.network.messages.response.response import Response
from src.network.messages.serialization.message_deserializer import MessageDeserializer
from src.network.messages.serialization.message_serializer import MessageSerializer
from src.network.network_config import NETWORK_MSG_LEN_FORMAT, NETWORK_SERVER_PORT

T = TypeVar("T", bound=Response)


class NetworkClient:
    def __init__(self, ip: str, message_serializer: MessageSerializer, message_deserializer: MessageDeserializer):
        """
        Initialize a network client

        Args:
            message_reader (MessageReader): Message reader
            message_serializer (MessageSerializer): Message serializer
            message_deserializer (MessageDeserializer): Message deserializer
        """
        self._ip = ip
        self._port = NETWORK_SERVER_PORT
        self._socket: socket.socket | None = None
        self._message_reader = None
        self._serializer = message_serializer
        self._deserializer = message_deserializer
        self._msg_len = NETWORK_MSG_LEN_FORMAT

    def connect(self) -> None:
        """
        Establishes a persistent connection to the server.
        """
        if self._socket:
            raise RuntimeError("Already connected")

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._ip, self._port))

        self._message_reader = StreamMessageReader(self._socket.makefile("rb"), self._msg_len)

    def send_request(self, request: Request) -> T:
        """
        Sends a request over the open socket and returns the response.

        Args:
            request (Request): The request to send.

        Returns:
            T: The deserialized response from the server.
        """
        if self._socket is None:
            raise RuntimeError("Not connected")

            # --- Serialize ---
        data = self._serializer.serialize(request)
        length = struct.pack(NETWORK_MSG_LEN_FORMAT, len(data))
        self._socket.sendall(length + data)

        msg_bytes = self._message_reader.read()
        return self._deserializer.deserialize(msg_bytes)


    def close(self) -> None:
        """
        Closes the persistent connection.
        """
        if self._socket:
            self._socket.close()
            self._socket = None
