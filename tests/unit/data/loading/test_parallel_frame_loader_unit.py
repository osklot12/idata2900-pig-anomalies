import pytest
from unittest.mock import MagicMock, call, patch
from src.data.loading.parallel_frame_loader import ParallelFrameLoader
from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback function."""
    return MagicMock()

@pytest.fixture
def dummy_gcp_data_loader():
    """Fixture to provide a dummy GCP data loader."""
    return DummyGCPDataLoader(bucket_name="dummy_bucket", credentials_path="fake/path")

@pytest.fixture
def video_list():
    """Fixture to provide a list of video names."""
    return ["video1.mp4", "video2.mp4"]

@pytest.fixture
def frame_shape():
    """Fixture to provide frame shape."""
    return (224, 224, 3)  # Example shape (H, W, C)
