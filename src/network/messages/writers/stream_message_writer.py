import struct
from io import BufferedWriter

from src.network.messages.writers.message_writer import MessageWriter


class StreamMessageWriter(MessageWriter):
    """A message writer for writing to a streams."""

    def __init__(self, stream: BufferedWriter, len_format: str):
        """
        Initializes a StreamMessageWriter instance.

        Args:
            stream (BufferedWriter): the streams to write to
            len_format (str): the length of the message to write
        """
        self._stream = stream
        self._len_format = len_format

    def write(self, message: bytes) -> None:
        length_prefix = struct.pack(self._len_format, len(message))
        self._stream.write(length_prefix + message)
        self._stream.flush()