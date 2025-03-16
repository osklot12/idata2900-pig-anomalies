from unittest.mock import patch, MagicMock
from requests.exceptions import HTTPError

import pytest

from src.data.loading.loaders.gcs_video_loader import GCSVideoLoader
from tests.utils.dummies.dummy_auth_service import DummyAuthService


@pytest.fixture
def gcs_video_loader():
    """Fixture to provide a GCSVideoLoader instance."""
    return GCSVideoLoader("test-bucket", DummyAuthService())

@patch.object(GCSVideoLoader, "_make_request")
def test_load_video_success(mock_make_request, gcs_video_loader):
    """Tests successful loading of video data."""
    # arrange
    video_id = "sample_video.mp4"
    mock_video_content = b'\x00\x01\x02\x03\x04'

    mock_response = MagicMock()
    mock_response.content = mock_video_content
    mock_make_request.return_value = mock_response

    # act
    result = gcs_video_loader.load_video(video_id)

    # assert
    mock_make_request.assert_called_once_with(gcs_video_loader._get_file_url(video_id))
    assert isinstance(result, bytearray)
    assert result == bytearray(mock_video_content)

@patch.object(GCSVideoLoader, "_make_request")
def test_load_video_not_found(mock_make_request, gcs_video_loader):
    """Tests handling when the requested video is not found."""
    # arrange
    video_id = "sample_video.mp4"
    error_msg = "404 Client Error: Not Found"

    mock_make_request.side_effect = HTTPError(error_msg)

    # act & assert
    with pytest.raises(HTTPError, match=error_msg):
        gcs_video_loader.load_video(video_id)

    mock_make_request.assert_called_once_with(gcs_video_loader._get_file_url(video_id))