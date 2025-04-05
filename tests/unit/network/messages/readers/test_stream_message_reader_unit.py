import struct
from unittest.mock import Mock

import pytest

from src.network.messages.readers.stream_message_reader import StreamMessageReader
from src.network.messages.readers.stream_read_error import StreamReadError


@pytest.fixture
def stream():
    """Fixture to provide a streams."""
    return Mock()


@pytest.mark.unit
def test_successful_read(stream):
    """Tests that read() successfully reads a valid streams."""
    # arrange
    stream.read.side_effect = [
        struct.pack(">I", 5),
        b"abcde"
    ]

    reader = StreamMessageReader(stream, ">I")

    # act
    message = reader.read()

    # assert
    assert message == b"abcde"
    assert stream.read.call_count == 2


@pytest.mark.unit
def test_raises_when_stream_returns_none(stream):
    """Tests that read() raises when the streams returns None."""
    # arrange
    stream.read.return_value = None

    reader = StreamMessageReader(stream, ">I")

    # act & assert
    with pytest.raises(StreamReadError, match="got None"):
        reader.read()


@pytest.mark.unit
def test_raises_when_stream_returns_too_few_bytes(stream):
    """Tests that read() raises when the streams returns too few bytes."""
    # arrange
    stream.read.side_effect = [
        struct.pack(">I", 5),
        b"abc"
    ]

    reader = StreamMessageReader(stream, ">I")

    # act & assert
    with pytest.raises(StreamReadError, match="Expected 5 bytes"):
        reader.read()
