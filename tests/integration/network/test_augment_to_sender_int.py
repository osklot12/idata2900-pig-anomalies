import numpy as np
from unittest.mock import MagicMock

from src.data.augment.combined_augmentor import CombinedAugmentation
from src.network.pipeline_ddp_sender import PipelineDataSender


def create_dummy_frame(shape=(640, 640, 3)):
    return np.random.randint(0, 256, size=shape, dtype=np.uint8)


def create_dummy_annotations():
    return [("pig", 100.0, 150.0, 50.0, 60.0)]


def annotation_tuples_to_dicts(annotation_list):
    """Convert (class, x, y, w, h) → dict format."""
    return [
        {"class": cls, "x": x, "y": y, "w": w, "h": h}
        for cls, x, y, w, h in annotation_list
    ]


def test_augmentor_output_sends_through_sender(monkeypatch):
    # Arrange
    dummy_frame = create_dummy_frame()
    annotations = create_dummy_annotations()

    augmentor = CombinedAugmentation(num_versions=2)
    augmented = augmentor.augment(dummy_frame, annotations)

    frames = []
    annots = []

    for frame, annotation_list in augmented:
        frames.append(frame)
        annots.append(annotation_tuples_to_dicts(annotation_list))

    # Create sender with mock IPs
    sender = PipelineDataSender(worker_ips=["127.0.0.1"])

    # Mock socket dictionary
    mock_socket = MagicMock()
    sender.sockets["127.0.0.1"] = mock_socket

    # Act
    sender.send_data(frames, annots)

    # Assert
    assert mock_socket.sendall.call_count == len(frames)
