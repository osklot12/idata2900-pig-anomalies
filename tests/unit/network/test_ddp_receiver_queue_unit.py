import pytest
import pickle
import struct
import numpy as np
from multiprocessing import Queue
from unittest.mock import MagicMock, patch
from src.worker.pipeline_ddp_receiver import PipelineDataReceiver


@pytest.fixture
def receiver():
    """Creates a receiver instance."""
    return PipelineDataReceiver(host="127.0.0.1", port=50051)

def test_queue_sync_single_client(receiver):
    """Test if the receiver correctly queues data from a single client."""
    queue = Queue()

    test_frames = [
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
        np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
    ]
    test_annotations = [
        [{"label": "test1", "bbox": [10, 10, 50, 50]}],
        [{"label": "test2", "bbox": [20, 20, 60, 60]}],
    ]

    for i in range(2):
        data = pickle.dumps({"frame": test_frames[i], "annotations": test_annotations[i]})
        message = struct.pack(">L", len(data)) + data

        with patch("socket.socket") as mock_socket:
            mock_conn = MagicMock()
            mock_socket.return_value = mock_conn
            mock_conn.recv.side_effect = [message[:4], message[4:]]

            received_data = receiver.receive_data(mock_conn)
            if received_data:
                queue.put((received_data["frame"], received_data["annotations"]))

    assert queue.qsize() == 2  # Ensure both items were added

    # Verify correct order
    for i in range(2):
        frame, annotations = queue.get()
        assert (frame == test_frames[i]).all()  # Check frame content
        assert annotations[0]["label"] == test_annotations[i][0]["label"]  # Check labels

def test_queue_sync_multiple_clients(receiver):
    """Test if the queue handles multiple clients correctly."""
    queue = Queue()

    test_data = [
        {"frame": np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
         "annotations": [{"label": "client1", "bbox": [5, 5, 40, 40]}]},
        {"frame": np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8),
         "annotations": [{"label": "client2", "bbox": [15, 15, 50, 50]}]},
    ]

    with patch("socket.socket") as mock_socket:
        client_sockets = [MagicMock(), MagicMock()]
        mock_socket.side_effect = client_sockets

        for i, mock_conn in enumerate(client_sockets):
            data = pickle.dumps(test_data[i])
            message = struct.pack(">L", len(data)) + data
            mock_conn.recv.side_effect = [message[:4], message[4:]]

            received_data = receiver.receive_data(mock_conn)
            if received_data:
                queue.put((received_data["frame"], received_data["annotations"]))

    assert queue.qsize() == 2  # Ensure data from both clients is added

    # Verify correct processing order
    for i in range(2):
        frame, annotations = queue.get()
        assert (frame == test_data[i]["frame"]).all()
        assert annotations[0]["label"] == test_data[i]["annotations"][0]["label"]
