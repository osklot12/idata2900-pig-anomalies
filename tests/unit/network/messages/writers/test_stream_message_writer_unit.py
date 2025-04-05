import struct
from unittest.mock import Mock

import pytest

from src.network.messages.writers.stream_message_writer import StreamMessageWriter


@pytest.fixture
def stream():
    """Fixture to provide a streams."""
    return Mock()


def test_write_writes_length_prefixed_message(stream):
    """Tests that write() writes a length prefixed message successfully."""
    # arrange
    writer = StreamMessageWriter(stream, ">I")

    message = b"hello"

    # act
    writer.write(message)

    # assert
    expected_prefix = struct.pack(">I", len(message))
    expected_data = expected_prefix + message

    stream.write.assert_called_once_with(expected_data)
    stream.flush.assert_called_once()