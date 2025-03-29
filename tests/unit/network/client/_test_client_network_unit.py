import struct
import pytest
from unittest.mock import MagicMock, create_autospec, patch

from src.network.client.client_network import NetworkClient
from src.network.messages.requests.request import Request
from src.network.messages.responses.response import Response
from src.network.messages.serialization.message_serializer import MessageSerializer
from src.network.messages.serialization.message_deserializer import MessageDeserializer
from src.network.network_config import NETWORK_MSG_LEN_FORMAT


@pytest.fixture
def serializer():
    return create_autospec(MessageSerializer)


@pytest.fixture
def deserializer():
    return create_autospec(MessageDeserializer)


@pytest.fixture
def client(serializer, deserializer):
    return NetworkClient("127.0.0.1", serializer, deserializer)


@pytest.mark.unit
@patch("socket.socket")
@patch("src.network.client.client_network.StreamMessageReader")
def test_connect_initializes_socket_and_reader(mock_reader_class, mock_socket_class, client):
    mock_socket = MagicMock()
    mock_socket.makefile.return_value = MagicMock()
    mock_socket_class.return_value = mock_socket

    client.connect()

    mock_socket.connect.assert_called_once_with(("127.0.0.1", client._port))
    mock_reader_class.assert_called_once_with(mock_socket.makefile(), NETWORK_MSG_LEN_FORMAT)


@pytest.mark.unit
@patch("socket.socket")
@patch("src.network.client.client_network.StreamMessageReader")
def test_send_request_sends_and_receives(mock_reader_class, mock_socket_class, client, serializer, deserializer):
    # Setup socket + reader
    mock_sock = MagicMock()
    mock_sock.makefile.return_value = MagicMock()
    mock_socket_class.return_value = mock_sock

    mock_reader = MagicMock()
    mock_reader.read.return_value = b"responses-bytes"
    mock_reader_class.return_value = mock_reader

    # Fake connection
    client.connect()

    # Setup mock behavior
    request = create_autospec(Request)
    serializer.serialize.return_value = b"request-bytes"
    expected_length = struct.pack(NETWORK_MSG_LEN_FORMAT, len(b"request-bytes"))
    response = create_autospec(Response)
    deserializer.deserialize.return_value = response

    # Call
    result = client.send_request(request)

    # Assert send + receive
    mock_sock.sendall.assert_called_once_with(expected_length + b"request-bytes")
    mock_reader.read.assert_called_once()
    deserializer.deserialize.assert_called_once_with(b"responses-bytes")
    assert result == response

@pytest.mark.unit
@patch("socket.socket")
def test_close_closes_socket(mock_socket_class, client):
    mock_sock = MagicMock()
    mock_socket_class.return_value = mock_sock

    client._socket = mock_sock
    client.close()

    mock_sock.close.assert_called_once()
    assert client._socket is None
