import numpy as np
from unittest.mock import MagicMock
from src.worker.pipeline_ddp_receiver import PipelineDataReceiver
from src.worker.data_queue import data_queue

def test_receiver_tensor_integration(monkeypatch):
    receiver = PipelineDataReceiver()

    # Create fake frame and annotation
    mock_data = {
        "frame": (255 * np.ones((224, 224, 3))).astype(np.uint8),
        "annotations": [{"bbox": [0, 0, 50, 50], "label": 1}]
    }

    # Patch receive_data to return mock_data once, then None to break the loop
    receiver.receive_data = MagicMock(side_effect=[mock_data, None])

    # Patch client_socket to be a dummy (not actually used now)
    class DummySocket:
        def close(self):
            self.closed = True

    socket = DummySocket()

    # Run the handler — it should exit after one pass
    receiver.handle_client(socket)

    # Test results
    assert hasattr(socket, "closed")
    assert not data_queue.empty()

    image_tensor, target = data_queue.get()
    assert image_tensor.shape == (3, 224, 224)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].shape == (1,)
