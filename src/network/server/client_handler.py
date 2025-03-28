import pickle
import struct
from socket import socket

from src.network.messages.requests.request import Request
from src.network.network_config import NETWORK_MSG_LEN_FIELD_BYTES, NETWORK_MSG_LEN_FORMAT


class ClientHandler:
    """Handles incoming client requests."""

    def __init__(self, client_socket: socket):
        """
        Initializes a ClientHandler instance.

        Args:
            client_socket (socket.socket): the client socket
        """
        self._client_socket = client_socket

    def handle(self) -> None:
        """Handles the client request."""
        pass

    def _receive_msg(self) -> Request:
        msg_len = self._extract_len()
        return self._extract_data(msg_len)

    def _extract_len(self):
        """Extracts the message length."""
        raw_msg_len = self._read_bytes(NETWORK_MSG_LEN_FIELD_BYTES)
        if not raw_msg_len:
            raise RuntimeError("Could not read length field for request")
        msg_len = struct.unpack(NETWORK_MSG_LEN_FORMAT, raw_msg_len)[0]
        return msg_len

    def _extract_data(self, msg_len):
        raw_data = self._read_bytes(msg_len)
        if not raw_data:
            raise RuntimeError("Could not read data payload for request")
        return pickle.loads(raw_data)

    def _read_bytes(self, n_bytes: int) -> bytes:
        """Read exactly `n_bytes` bytes from the socket."""
        buffer = b''

        closed = False
        while len(buffer) < n_bytes and not closed:
            chunk = self._client_socket.recv(n_bytes - len(buffer))
            if not chunk:
                closed = True
            buffer += chunk

        return buffer