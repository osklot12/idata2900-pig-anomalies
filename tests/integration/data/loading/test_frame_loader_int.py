from unittest.mock import Mock

import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.loading.frame_loader import FrameLoader
from src.data.loading.gcp_data_loader import GCPDataLoader
from tests.conftest import get_test_path

TEST_CREDENTIALS_PATH = get_test_path(".secrets/service-account.json")
TEST_BUCKET = "norsvin-g2b-behavior-prediction"
TEST_VIDEO_PREFIX = "g2b_behaviour/images/"
TEST_VIDEO_NAME = "avd13_cam1_20220314072829_20220314073013_fps2.0.mp4"

@pytest.fixture(scope="module")
def gcp_data_loader():
    """Fixture to provide a GCPDataLoader instance."""
    auth_service = GCPAuthService(credentials_path=TEST_CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=TEST_BUCKET, auth_service=auth_service)

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()

@pytest.mark.integration
def test_load_frames(gcp_data_loader, mock_callback):
    """Integration test: Ensures that frames are loaded and processed correctly."""

    # arrange
    video_blob_name = f"{TEST_VIDEO_PREFIX}{TEST_VIDEO_NAME}"
    frame_loader = FrameLoader(
        data_loader=gcp_data_loader,
        callback=mock_callback,
        frame_shape=(1520, 2688)
    )

    # act
    frame_loader.load_frames(video_blob_name)
    frame_loader.wait_for_completion()

    # assert
    assert mock_callback.call_count > 1

    last_call = mock_callback.call_args_list[-1]
    _, last_frame_index, last_frame, is_complete = last_call[0]

    assert last_frame is None
    assert is_complete is True