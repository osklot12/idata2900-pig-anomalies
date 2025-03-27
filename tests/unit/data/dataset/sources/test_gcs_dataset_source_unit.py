from unittest.mock import patch, MagicMock

import pytest
from requests.exceptions import HTTPError

from src.data.dataset.sources.gcs_source_registry import GCSSourceRegistry
from tests.utils.dummies.dummy_auth_service import DummyAuthService


@pytest.fixture
def gcs_file_manager():
    """Fixture to provide a GCSFileManager instance."""
    return GCSSourceRegistry("test-bucket", DummyAuthService())


@pytest.mark.unit
@patch.object(GCSSourceRegistry, "_make_request")
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
    result = gcs_file_manager.get_source_ids()

    # assert
    mock_make_request.assert_called_once_with(
        f"https://www.googleapis.com/storage/v1/b/{gcs_file_manager._bucket_name}/o"
    )
    assert isinstance(result, set)
    assert result == {"video_1.mp4", "video_2.mp4", "annotation_1.json"}


@pytest.mark.unit
@patch.object(GCSSourceRegistry, "_make_request")
def test_list_files_empty_bucket(mock_make_request, gcs_file_manager):
    """Tests listing files when the GCS bucket is empty."""
    # arrange
    mock_response = MagicMock()
    mock_response.json.return_value = {}
    mock_make_request.return_value = mock_response

    # act
    result = gcs_file_manager.get_source_ids()

    # assert
    mock_make_request.assert_called_once()
    assert isinstance(result, set)
    assert result == set()


@pytest.mark.unit
@patch.object(GCSSourceRegistry, "_make_request")
def test_list_files_not_found(mock_make_request, gcs_file_manager):
    """Tests handling when the requested bucket is not found."""
    # arrange
    error_msg = "404 Client Error: Not Found"
    mock_make_request.side_effect = HTTPError(error_msg)

    # act & assert
    with pytest.raises(HTTPError, match=error_msg):
        gcs_file_manager.get_source_ids()

    mock_make_request.assert_called_once()
