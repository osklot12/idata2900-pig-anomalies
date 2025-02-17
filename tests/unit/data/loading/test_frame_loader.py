import pytest
import tensorflow as tf
from src.data.loading.frame_loader import FrameLoader
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def received_frames():
    """Fixture to store received frames."""
    return []

@pytest.fixture
def frame_callback(received_frames):
    """Callback function that captures frames received from FrameLoader."""
    def callback(video_name, frame_index, frame, is_last_frame):
        received_frames.append((video_name, frame_index, frame))
    return callback

def test_load_frames(received_frames, frame_callback):
    """Test loading frames using dummy classes"""

    # use the dummy implementations
    dummy_gcp_loader = DummyGCPDataLoader(bucket_name="mock-bucket", credentials_path="mock-credentials.json")
