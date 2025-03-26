import pytest
import pickle
import struct
import numpy as np
from unittest.mock import patch, MagicMock
from multiprocessing import Queue
from src.worker.pipeline_ddp_receiver import PipelineDataReceiver

@pytest.fixture
def receiver():
    """Creates a receiver instance."""
    return PipelineDataReceiver(host="127.0.0.1", port=50051)

def test_receiver_accepts_connections(receiver):
    """Test if the receiver correctly binds and listens for connections."""
    with patch("socket.socket") as mock_socket:
        mock_server = MagicMock()
        mock_socket.return_value = mock_server

        receiver.server_socket = mock_server  # Manually set mocked socket
        receiver.server_socket.bind(("127.0.0.1", 50051))
        receiver.server_socket.listen(5)

        assert mock_server.bind.called
        assert mock_server.listen.called

def test_receiver_receives_data(receiver):
    """Test if receiver correctly gets data."""
    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    test_annotations = [{"label": "test", "bbox": [0, 0, 50, 50]}]

    data = pickle.dumps({"frame": test_frame, "annotations": test_annotations})
    message = struct.pack(">L", len(data)) + data

    with patch("socket.socket") as mock_socket:
        mock_conn = MagicMock()
        mock_socket.return_value = mock_conn
        mock_conn.recv.side_effect = [message[:4], message[4:]]

        received_data = receiver.receive_data(mock_conn)

        assert received_data is not None
        assert received_data["frame"].shape == (640, 640, 3)
        assert received_data["annotations"][0]["label"] == "test"

def test_receiver_puts_data_in_queue(receiver):
    """Test if receiver correctly puts received data in queue."""
    queue = Queue()

    test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    test_annotations = [{"label": "test", "bbox": [0, 0, 50, 50]}]

    data = pickle.dumps({"frame": test_frame, "annotations": test_annotations})
    message = struct.pack(">L", len(data)) + data

    with patch("socket.socket") as mock_socket:
        mock_conn = MagicMock()
        mock_socket.return_value = mock_conn
        mock_conn.recv.side_effect = [message[:4], message[4:]]

        received_data = receiver.receive_data(mock_conn)

        if received_data:
            queue.put((received_data["frame"], received_data["annotations"]))

        assert not queue.empty()
        frame, annotations = queue.get()
        assert frame.shape == (640, 640, 3)
        assert annotations[0]["label"] == "test"
