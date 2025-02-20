import pytest
from unittest.mock import Mock

from src.data.loading.frame_loader import FrameLoader

from tests.utils.dummies.dummy_gcp_data_loader import DummyGCPDataLoader


@pytest.fixture
def dummy_data_loader():
    """Fixture to provide a dummy GCPDataLoader."""
    return DummyGCPDataLoader(bucket_name="test-bucket", credentials_path="fake-path")


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()

def test_load_frames(dummy_data_loader, mock_callback):
    """Test that FrameLoader correctly loads and processes video frames."""

    # arrange
    video_blob_name = "test-video.mp4"
    frame_loader = FrameLoader(
        data_loader=dummy_data_loader,
        callback=mock_callback,
    )

    # act
    frame_loader.load_frames(video_blob_name)

    # assert
    assert mock_callback.call_count > 1

    # extract the last callback call arguments (last call should have `None` as frame)
    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_frame, is_complete = last_call[0]

    # the last call should indicate completion
    assert last_frame is None
    assert is_complete is True
