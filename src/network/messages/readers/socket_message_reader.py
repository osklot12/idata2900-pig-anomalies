import struct

from src.network.messages.readers.message_reader import MessageReader
from socket import socket

class SocketMessageReader(MessageReader):
    """A simple message reader."""

    def __init__(self, sock: socket, len_format: str, len_bytes: int):
        """
        Initializes a SocketMessageReader instance.

        Args:
            sock (socket): the socket to read from
            len_format (str): the length-prefixed framing format
            len_bytes (int): the length of the length-prefix
        """
        self._socket = sock
        self._len_format = len_format
        self._len_bytes = len_bytes

    def read(self) -> bytes:
        msg_len = self._extract_len()
        return self._extract_data(msg_len)

    def _extract_len(self):
        """Extracts the message length."""
        raw_msg_len = self._read_bytes(self._len_bytes)
        if not raw_msg_len:
            raise RuntimeError("Could not read length field for request")
        msg_len = struct.unpack(self._len_format, raw_msg_len)[0]
        return msg_len

    def _extract_data(self, msg_len):
        raw_data = self._read_bytes(msg_len)
        if not raw_data:
            raise RuntimeError("Could not read data payload for request")
        return raw_data

    def _read_bytes(self, n_bytes: int) -> bytes:
        """Read exactly `n_bytes` bytes from the socket."""
        buffer = b''

        while len(buffer) < n_bytes:
            chunk = self._socket.recv(n_bytes - len(buffer))
            if not chunk:
                raise ConnectionError(f"Expected {n_bytes} bytes but got {len(chunk)}")
            buffer += chunk

        return buffer