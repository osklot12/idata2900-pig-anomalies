import pytest
import socket
import pickle
import struct
import numpy as np
from unittest.mock import MagicMock, patch
from src.network.pipeline_ddp_sender import PipelineDataSender

@pytest.fixture
def sender():
    """Create a PipelineDataSender instance with a mock worker."""
    return PipelineDataSender(worker_ips=["127.0.0.1"], port=50051)

def test_sender_connects_to_workers(sender):
    """Test if the sender connects to workers properly."""
    with patch("socket.socket") as mock_socket:
        mock_conn = MagicMock()
        mock_socket.return_value = mock_conn

        sender.connect_to_workers()

        assert "127.0.0.1" in sender.sockets
        assert mock_conn.connect.called

def test_sender_serialization(sender):
    """Test if sender correctly serializes data."""
    frames = [np.zeros((640, 640, 3), dtype=np.uint8)]
    annotations = [[{"label": "test", "bbox": [0, 0, 50, 50]}]]

    with patch("socket.socket") as mock_socket:
        mock_conn = MagicMock()
        mock_socket.return_value = mock_conn

        sender.connect_to_workers()
        sender.send_data(frames, annotations)

        assert mock_conn.sendall.called

        sent_data = mock_conn.sendall.call_args[0][0]
        message_size = struct.unpack(">L", sent_data[:4])[0]
        assert message_size > 0  # Ensures data is not empty

        data = pickle.loads(sent_data[4:])
        assert data["frame"].shape == (640, 640, 3)
        assert data["annotations"][0]["label"] == "test"

def test_sender_sends_to_multiple_workers():
    """Test sender handling multiple worker connections."""
    sender = PipelineDataSender(worker_ips=["127.0.0.1", "127.0.0.2"], port=50051)

    with patch("socket.socket") as mock_socket:
        mock_conn = MagicMock()
        mock_socket.return_value = mock_conn

        sender.connect_to_workers()
        assert len(sender.sockets) == 2  # Ensure both workers are connected
