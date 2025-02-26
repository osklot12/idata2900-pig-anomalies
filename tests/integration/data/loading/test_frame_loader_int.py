from unittest.mock import Mock

import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.loading.frame_loader import FrameLoader
from src.data.loading.gcp_data_loader import GCPDataLoader
from src.utils.norsvin_bucket_parser import NorsvinBucketParser
from tests.utils.constants.sample_bucket_files import SampleBucketFiles


@pytest.fixture(scope="module")
def gcp_data_loader():
    """Fixture to provide a GCPDataLoader instance."""
    auth_service = GCPAuthService(credentials_path=NorsvinBucketParser.CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=NorsvinBucketParser.BUCKET_NAME, auth_service=auth_service)

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return Mock()

@pytest.mark.integration
def test_load_frames(gcp_data_loader, mock_callback):
    """Integration test: Ensures that frames are loaded and processed correctly."""

    # arrange
    video_blob_name = NorsvinBucketParser.get_video_blob_name(SampleBucketFiles.SAMPLE_VIDEO_FILE)
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