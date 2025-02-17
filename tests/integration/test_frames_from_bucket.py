import pytest
from src.data.loading.gcp_data_loader import GCPDataLoader
from cppbindings import FrameStream

TEST_BUCKET = "norsvin-g2b-behavior-prediction"
TEST_CREDENTIALS_PATH = "../../.secrets/service-account.json"
TEST_VIDEO_PREFIX = "g2b_behaviour/images/"
TEST_VIDEO_NAME = "avd13_cam1_20220314072829_20220314073013_fps2.0.mp4"

@pytest.fixture
def gcp_loader():
    """Create a real GCPDataLoader instance with actual credentials."""
    return GCPDataLoader(bucket_name=TEST_BUCKET, credentials_path=TEST_CREDENTIALS_PATH)

def test_framestream_with_gcpdataloader(gcp_loader):
    """Tests if FrameStream can read frames from a video streamed by GCPDataLoader."""
    video_blob_name = f"{TEST_VIDEO_PREFIX}{TEST_VIDEO_NAME}"
    video_data = gcp_loader.download_video(video_blob_name)

    video_bytes = video_data.getvalue()
    fstream = FrameStream(bytearray(video_bytes))

    while fstream.read():
        print("Frame read")