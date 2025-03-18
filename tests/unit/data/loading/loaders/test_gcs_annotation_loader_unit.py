from unittest.mock import patch, MagicMock

import pytest
from requests.exceptions import HTTPError

from src.data.loading.loaders.gcs_annotation_loader import GCSAnnotationLoader
from tests.utils.dummies.dummy_auth_service import DummyAuthService


@pytest.fixture
def gcs_annotation_loader():
    """Fixture to provide a GCSAnnotationLoader instance."""
    return GCSAnnotationLoader("test-bucket", DummyAuthService())

@patch.object(GCSAnnotationLoader, "_make_request")
def test_load_annotation_success(mock_make_request, gcs_annotation_loader):
    """Tests successful loading of an annotation file."""
    # arrange
    annotation_id = "sample_annotation.json"
    mock_json_data = {"class": "anomaly_behavior", "frame": 42}

    mock_response = MagicMock()
    mock_response.json.return_value = mock_json_data
    mock_make_request.return_value = mock_response

    # act
    result = gcs_annotation_loader.load_video_annotations(annotation_id)

    # assert
    mock_make_request.assert_called_once_with(gcs_annotation_loader._get_file_url(annotation_id))
    assert isinstance(result, dict)
    assert result == mock_json_data

@patch.object(GCSAnnotationLoader, "_make_request")
def test_load_annotation_not_found(mock_make_request, gcs_annotation_loader):
    """Tests handling when the requested annotation file is not found."""
    # arrange
    annotation_id = "missing_annotation.json"
    error_msg = "404 Client Error: Not Found"

    mock_make_request.side_effect = HTTPError(error_msg)

    # act & assert
    with pytest.raises(HTTPError, match=error_msg):
        gcs_annotation_loader.load_video_annotations(annotation_id)

    mock_make_request.assert_called_once_with(gcs_annotation_loader._get_file_url(annotation_id))