import os
import pytest
import json
import requests
from unittest.mock import patch, MagicMock
from io import BytesIO
from src.data.data_loader import GCPStorageHandler, VideoAnnotationLoader

# 🔹 Configure test constants
TEST_BUCKET = "test-bucket"
TEST_CREDENTIALS_PATH = "tests/mock_credentials.json"


@pytest.fixture
def storage_handler():
    """Fixture to create a GCPStorageHandler instance with mocked credentials."""
    handler = GCPStorageHandler(bucket_name=TEST_BUCKET, credentials_path=TEST_CREDENTIALS_PATH)

    with patch.object(handler, "credentials_path", TEST_CREDENTIALS_PATH), \
            patch("os.path.exists", return_value=True), \
            patch("builtins.open", new_callable=MagicMock), \
            patch("src.data.data_loader.service_account.Credentials.from_service_account_file") as mock_creds:
        mock_creds.return_value = MagicMock()
        mock_creds.return_value.token = "mock_access_token"

        yield handler



@pytest.fixture
def video_loader(storage_handler):
    """Fixture to create an instance of VideoAnnotationLoader for tests."""
    return VideoAnnotationLoader(storage_handler)


# 🔹 Test Authentication
@patch("src.data.data_loader.service_account.Credentials.from_service_account_file")
@patch("os.path.exists", return_value=True)  # 🔹 Pretend the file always exists
@patch("builtins.open", new_callable=MagicMock)  # 🔹 Mock file opening
def test_authentication(mock_open, mock_exists, mock_creds, storage_handler):
    """Test if the service account retrieves an access token correctly."""
    mock_creds.return_value = MagicMock()
    mock_creds.return_value.token = "mock_access_token"

    storage_handler.authenticate()

    assert storage_handler.token == "mock_access_token"




# 🔹 Test Listing Files in Bucket
@patch("requests.get")
def test_list_files(mock_get, storage_handler):
    """Test if listing files in GCS works."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {
        "items": [{"name": "test_video.mp4"}, {"name": "test_annotation.json"}]
    }

    files = storage_handler.list_files(file_extension=".mp4")

    assert len(files) == 1
    assert files[0] == "test_video.mp4"


# 🔹 Test Streaming a Video File
@patch("requests.get")
def test_stream_video(mock_get, storage_handler):
    """Test if video streaming returns a valid BytesIO object."""
    mock_get.return_value.status_code = 200
    mock_get.return_value.content = b"mock binary data"

    video_stream = storage_handler.stream_video("test_video.mp4")

    assert isinstance(video_stream, BytesIO)
    assert video_stream.read() == b"mock binary data"


# 🔹 Test Downloading JSON Annotation
@patch("requests.get")
def test_download_json(mock_get, storage_handler):
    """Test if JSON annotations are correctly downloaded and parsed."""
    mock_json_data = {"annotation": "mock_data"}
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = mock_json_data

    annotation_data = storage_handler.download_json("test_annotation.json")

    assert annotation_data == mock_json_data


# 🔹 Test Video-Annotation Matching
def test_video_annotation_matching(video_loader):
    """Test if videos correctly pair with their annotations."""
    video_loader.storage_handler.list_files = MagicMock(
        side_effect=[
            ["video1.mp4", "video2.mp4"],  # Videos
            ["g2b_behaviour/releases/g2b-prediction/video1.json"],  # Annotations
        ]
    )

    video_loader.storage_handler.stream_video = MagicMock(return_value=BytesIO(b"mock video data"))
    video_loader.storage_handler.download_json = MagicMock(return_value={"mock_annotation": True})

    combined_data = video_loader.load_videos_with_annotations()

    assert len(combined_data) == 1
    assert "video1.mp4" in combined_data
    assert combined_data["video1.mp4"]["annotation"] == {"mock_annotation": True}
