import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from src.data.gcp_data_loader import GCPDataLoader

# Constants for testing
TEST_BUCKET = "test-bucket"
TEST_CREDENTIALS_PATH = "../../data/test-account.json"
TEST_VIDEO_NAME = "test_video.mp4"
TEST_JSON_NAME = "test_annotation.json"

@pytest.fixture
def gcp_loader():
    """Fixture to create a GCPDataLoader instance with mocked credentials."""
    with patch("src.data.gcp_data_loader.service_account.Credentials.from_service_account_file") as mock_creds:
        mock_creds.return_value = MagicMock()
        mock_creds.return_value.token = "mock_access_token"
        mock_creds.return_value.valid = True

        loader = GCPDataLoader(bucket_name=TEST_BUCKET, credentials_path=TEST_CREDENTIALS_PATH)
        loader.creds = mock_creds.return_value

        yield loader

def test_authentication(gcp_loader):
    """Tests that authentication works correctly."""
    assert gcp_loader.creds.token == "mock_access_token"


@patch("src.data.gcp_data_loader.service_account.Credentials.from_service_account_file")
def test_refresh_token(mock_creds, gcp_loader):
    """Tests that refresh_token updates the access token."""
    mock_creds_instance = MagicMock()
    mock_creds_instance.valid = False
    mock_creds_instance.token = "new_mock_token"
    mock_creds_instance.refresh = MagicMock(side_effect=lambda req: setattr(mock_creds_instance, "valid", True))

    mock_creds.return_value = mock_creds_instance
    gcp_loader.creds = mock_creds.return_value

    gcp_loader.refresh_token()
    assert gcp_loader.creds.valid  # Now should be True
    assert gcp_loader.creds.token == "new_mock_token"

def test_get_access_token(gcp_loader):
    """Tests if get_access_token returns a valid token."""
    gcp_loader.creds.valid = False
    token = gcp_loader.get_access_token()
    assert token == gcp_loader.creds.token

@patch("requests.get")
def test_fetch_all_files(mock_get, gcp_loader):
    """Tests if fetch_all_files retrieves file names from GCS with caching."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"items": [{"name": "file1.mp4"}, {"name": "file2.json"}]}

    files = gcp_loader.fetch_all_files()
    assert len(files) == 2
    assert "file1.mp4" in files
    assert "file2.json" in files

    # Ensure caching works
    cached_files = gcp_loader.fetch_all_files()
    assert cached_files is files  # Same reference, cache is working

@patch("requests.get")
def test_list_files(mock_get, gcp_loader):
    """Test if list_files correctly filters files by prefix and extension."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "items": [
            {"name": "prefix_file1.mp4"},
            {"name": "prefix_file2.json"},
            {"name": "other_file3.mp4"}
        ]
    }

    files = gcp_loader.list_files(prefix="prefix", file_extension=".mp4")
    assert len(files) == 1
    assert files[0] == "prefix_file1.mp4"

@patch("requests.get")
def test_stream_video(mock_get, gcp_loader):
    """Test if stream_video returns a valid BytesIO object."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.iter_content = MagicMock(return_value=[b"chunk1", b"chunk2"])

    video_stream = gcp_loader.stream_video(TEST_VIDEO_NAME)
    assert isinstance(video_stream, BytesIO)
    assert video_stream.read() == b"chunk1chunk2"

@patch("requests.get")
def test_download_json(mock_get, gcp_loader):
    """Test if download_json correctly parses and returns JSON data."""
    mock_json_data = {"key": "value"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_json_data

    json_data = gcp_loader.download_json(TEST_JSON_NAME)
    assert json_data == mock_json_data

@patch("requests.get")
def test_download_json_error(mock_get, gcp_loader):
    """Test if download_json raises an exception on API error."""
    mock_get.return_value.status_code = 500
    mock_get.return_value.text = "Internal Server Error"

    with pytest.raises(RuntimeError, match="Failed to download"):
        gcp_loader.download_json(TEST_JSON_NAME)

@patch("requests.get")
def test_list_files_with_no_results(mock_get, gcp_loader):
    """Test if list_files handles empty responses correctly."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"items": []}  # No files found

    files = gcp_loader.list_files(prefix="missing_prefix", file_extension=".mp4")
    assert files == []  # Expect an empty list

@patch("requests.get")
def test_fetch_all_files_error(mock_get, gcp_loader):
    """Test if fetch_all_files raises an exception on API error."""
    mock_get.return_value.status_code = 500
    mock_get.return_value.text = "Internal Server Error"

    with pytest.raises(RuntimeError, match="Failed to fetch files"):
        gcp_loader.fetch_all_files()
