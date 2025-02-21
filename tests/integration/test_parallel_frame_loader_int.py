import pytest

from src.auth.gcp_auth_service import GCPAuthService
from src.data.loading.gcp_data_loader import GCPDataLoader
from src.data.loading.parallel_frame_loader import ParallelFrameLoader
from tests.conftest import get_test_path

TEST_CREDENTIALS_PATH = get_test_path(".secrets/service-account.json")
TEST_BUCKET = "norsvin-g2b-behavior-prediction"
TEST_VIDEO_PREFIX = "g2b_behaviour/images/"

@pytest.fixture
def gcp_data_loader():
    """Fixture to provide a GCPDataLoader instance."""
    auth_service = GCPAuthService(credentials_path=TEST_CREDENTIALS_PATH)
    return GCPDataLoader(bucket_name=TEST_BUCKET, auth_service=auth_service)

@pytest.fixture
def callback():
    """Fixture to track callback calls in a list."""
    events = []

    def callback(video_name, frame_index, frame_data, is_last):
        """Callback that records all frame processing events."""
        events.append((video_name, frame_index, frame_data is not None, is_last))

    return callback, events

def get_video_blob_name(video_name):
    """Returns the proper video blob name for a given video name."""
    return f"{TEST_VIDEO_PREFIX}{video_name}"

@pytest.mark.integration
def test_load_frames_parallel(gcp_data_loader, callback):
    """Integration test: Ensures that frames are loaded and processed correctly in parallel."""

    # arrange
    callback, events = callback
    video_list = [
        get_video_blob_name("avd13_cam1_20220314072829_20220314073013_fps2.0.mp4"),
        get_video_blob_name("avd13_cam1_20220314073029_20220314073258_fps2.0.mp4"),
        get_video_blob_name("avd13_cam1_20220314073329_20220314073458_fps2.0.mp4")
    ]

    parallel_loader = ParallelFrameLoader(gcp_data_loader, callback, num_threads=2)

    # act
    parallel_loader.load_frames(video_list)

    # assert
    processed_videos = {event[0] for event in events}
    assert processed_videos == set(video_list), "Not all videos processed."

    for video in video_list:
        assert any(event[0] == video and event[2] for event in events), f"No frames processed for video {video}."

    for video in video_list:
        assert any(event[0] == video and event[3] for event in events), f"Completion signal missing for video {video}."