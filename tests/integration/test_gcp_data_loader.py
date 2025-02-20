import pytest
import os
from io import BytesIO
from src.data.loading.gcp_data_loader import GCPDataLoader
from src.auth.gcp_auth_service import GCPAuthService

# constants
TEST_BUCKET = "norsvin-g2b-behavior-prediction"
TEST_CREDENTIALS_PATH = "../../.secrets/service-account.json"
TEST_VIDEO_PREFIX = "g2b_behaviour/images/"
TEST_VIDEO_NAME = "avd13_cam1_20220314072829_20220314073013_fps2.0.mp4"
TEST_JSON_PREFIX = "g2b_behaviour/releases/g2b-prediction/annotations/"
TEST_JSON_NAME = "avd13_cam1_20220314072829_20220314073013_fps2.0.json"

@pytest.fixture(scope="module")
def gcp_loader():
    """Creates a GCPDataLoader instance using real credentials for integration testing."""
    auth_service = GCPAuthService(credentials_path=TEST_CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=TEST_BUCKET, auth_service=auth_service)

@pytest.mark.integration
def test_fetch_all_files(gcp_loader):
    """Integration test: Checks if we can fetch real files from GCS."""
    files = gcp_loader.fetch_all_files()
    assert isinstance(files, list)

@pytest.mark.integration
def test_list_video_files(gcp_loader):
    """Integration test: Lists only video files from the video directory."""
    video_files = gcp_loader.list_files(prefix=TEST_VIDEO_PREFIX, file_extension=".mp4")
    assert isinstance(video_files, list)
    assert len(video_files) > 0
    assert all(f.startswith(TEST_VIDEO_PREFIX) and f.endswith(".mp4") for f in video_files)

@pytest.mark.integration
def test_list_annotation_files(gcp_loader):
    """Integration test: Lists only annotation files from the annotation directory."""
    annotation_files = gcp_loader.list_files(prefix=TEST_JSON_PREFIX, file_extension=".json")
    assert isinstance(annotation_files, list)
    assert len(annotation_files) > 0
    assert all(f.startswith(TEST_JSON_PREFIX) and f.endswith(".json") for f in annotation_files)

@pytest.mark.integration
def test_download_video(gcp_loader):
    """Integration test: Downloads a real video file from GCS."""
    video_data = gcp_loader.download_video(f"{TEST_VIDEO_PREFIX}{TEST_VIDEO_NAME}")
    assert isinstance(video_data, bytearray)
    assert len(video_data) > 0

def test_download_json(gcp_loader):
    """Integration test: Downloads and parses a real JSON file from GCS."""
    json_data = gcp_loader.download_json(f"{TEST_JSON_PREFIX}{TEST_JSON_NAME}")
    assert isinstance(json_data, dict)
    assert len(json_data) > 0

def test_handle_missing_file(gcp_loader):
    """Integration test: Ensures proper error handling for a non-existent file."""
    with pytest.raises(RuntimeError, match="Failed to download"):
        gcp_loader.download_json("non_existent_file.mp4")