import pytest
from unittest.mock import patch
from src.data.loading.gcp_data_loader import GCPDataLoader
from tests.utils.dummies.dummy_gcp_auth_service import DummyGCPAuthService

# Constants for testing
TEST_BUCKET = "test-bucket"
TEST_VIDEO_NAME = "test_video.mp4"
TEST_JSON_NAME = "test_annotation.json"

@pytest.fixture
def gcp_loader():
    """Creates a GCPDataLoader instance with dummy auth service."""
    return GCPDataLoader(bucket_name=TEST_BUCKET, auth_service=DummyGCPAuthService())

@patch("requests.get")
def test_fetch_all_files(mock_get, gcp_loader):
    """Mocks the API call to Google Cloud Storage and tests fetch_all_files."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"items": [{"name": "file1.mp4"}, {"name": "file2.json"}]}

    files = gcp_loader.fetch_all_files()
    assert files == ["file1.mp4", "file2.json"]

@patch("requests.get")
def test_list_files(mock_get, gcp_loader):
    """Tests if list_files correctly filters files bby prefix and extension."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "items": [
            {"name": "g2b_behaviour/images/video1.mp4"},
            {"name": "g2b_behaviour/images/video2.mp4"},
            {"name": "g2b_behaviour/annotations/annotation1.json"},
            {"name": "g2b_behaviour/annotations/annotation2.json"}
        ]
    }

    video_files = gcp_loader.list_files(prefix="g2b_behaviour/images/", file_extension=".mp4")
    annotation_files = gcp_loader.list_files(prefix="g2b_behaviour/annotations/", file_extension=".json")

    assert video_files == ["g2b_behaviour/images/video1.mp4", "g2b_behaviour/images/video2.mp4"]
    assert annotation_files == ["g2b_behaviour/annotations/annotation1.json",
                                "g2b_behaviour/annotations/annotation2.json"]

@patch("requests.get")
def test_download_video(mock_get, gcp_loader):
    """Mocks the API call and tests if download_video correctly returns a BytesIO object."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"chunk1chunk2"

    video_stream = gcp_loader.download_video(TEST_VIDEO_NAME)

    assert isinstance(video_stream, bytearray)
    mock_get.assert_called_once_with(
        f"https://storage.googleapis.com/{TEST_BUCKET}/{TEST_VIDEO_NAME}",
        headers={"Authorization": "Bearer dummy_access_token"},
        stream=True,
        timeout=60
    )

@patch("requests.get")
def test_download_json(mock_get, gcp_loader):
    """Mocks API response for JSON download."""
    mock_json_data = {"key": "value"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_json_data

    json_data = gcp_loader.download_json(TEST_JSON_NAME)
    assert json_data == mock_json_data

@patch("requests.get")
def test_download_json_error(mock_get, gcp_loader):
    """Tests if download_json raises an exception on an API error."""
    mock_get.return_value.status_code = 500
    mock_get.return_value.text = "Internal Server Error"

    with pytest.raises(RuntimeError, match="Failed to download"):
        gcp_loader.download_json(TEST_JSON_NAME)