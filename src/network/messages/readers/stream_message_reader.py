import struct
from io import BufferedReader

from src.network.messages.readers.stream_read_error import StreamReadError
from src.network.messages.readers.message_reader import MessageReader

class StreamMessageReader(MessageReader):
    """A message reader reading from a stream."""

    def __init__(self, stream: BufferedReader, len_format: str):
        """
        Initializes a SocketMessageReader instance.

        Args:
            stream (BufferedReader): the stream to read from
            len_format (str): the length-prefixed framing format
        """
        self._stream = stream
        self._len_format = len_format
        self._len_bytes = struct.calcsize(len_format)

    def read(self) -> bytes:
        length = self._read_bytes(self._len_bytes)
        msg_len = struct.unpack(self._len_format, length)[0]
        return self._read_bytes(msg_len)

    def _read_bytes(self, n: int) -> bytes:
        data = bytearray()
        while len(data) < n:
            chunk = self._stream.read(n - len(data))
            if not chunk:
                raise StreamReadError(f"Expected {n} bytes, got {len(data)} before stream closed")
            data.extend(chunk)
        return bytes(data)