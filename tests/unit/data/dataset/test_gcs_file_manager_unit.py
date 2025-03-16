from unittest.mock import patch, MagicMock

import pytest
from requests.exceptions import HTTPError

from src.data.dataset.gcs_file_manager import GCSFileManager
from tests.utils.dummies.dummy_gcp_auth_service import DummyGCPAuthService


@pytest.fixture
def gcs_file_manager():
    """Fixture to provide a GCSFileManager instance."""
    return GCSFileManager("test-bucket", DummyGCPAuthService())

@patch.object(GCSFileManager, "_make_request")
def test_list_files_success(mock_make_request, gcs_file_manager):
    """Tests successful listing of files in the GCS bucket."""
    # arrange
    mock_json_data = {
        "items": [
            {"name": "video_1.mp4"},
            {"name": "video_2.mp4"},
            {"name": "annotation_1.json"}
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = mock_json_data
    mock_make_request.return_value = mock_response

    # act
    result = gcs_file_manager.list_files()

    # assert
    mock_make_request.assert_called_once_with(
        f"https://www.googleapis.com/storage/v1/b/{gcs_file_manager._bucket_name}/o"
    )
    assert isinstance(result, list)
    assert result == ["video_1.mp4", "video_2.mp4", "annotation_1.json"]

@patch.object(GCSFileManager, "_make_request")
def test_list_files_empty_bucket(mock_make_request, gcs_file_manager):
    """Tests listing files when the GCS bucket is empty."""
    # arrange
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_make_request.return_value = mock_response

    # act
    result = gcs_file_manager.list_files()

    # assert
    mock_make_request.assert_called_once()
    assert isinstance(result, list)
    assert result == []

@patch.object(GCSFileManager, "_make_request")
def test_list_files_not_found(mock_make_request, gcs_file_manager):
    """Tests handling when the requested bucket is not found."""
    # arrange
    error_msg = "404 Client Error: Not Found"
    mock_make_request.side_effect = HTTPError(error_msg)

    # act & assert
    with pytest.raises(HTTPError, match=error_msg):
        gcs_file_manager.list_files()

    mock_make_request.assert_called_once()