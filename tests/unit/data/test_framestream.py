import pytest
import cppbindings

from tests.conftest import get_test_path

TEST_VIDEO_PATH = get_test_path("tests/data/sample-5s.mp4")

@pytest.fixture
def video_data():
    """Loads raw MP4 data from the test video file."""
    with open(TEST_VIDEO_PATH, "rb") as f:
        return list(f.read())


def test_framestream_initialization(video_data):
    """Test if FrameStream initializes correctly with actual video data."""
    fstream = cppbindings.FrameStream(video_data)
    assert fstream is not None, "FrameStream failed to initialize with real video data"


def test_framestream_read(video_data):
    """Test if FrameStream can read frames from actual video data."""
    fstream = cppbindings.FrameStream(video_data)

    frame = fstream.read()

    assert isinstance(frame, list), "FrameStream.read() should return a list"
    if frame:
        assert len(frame) > 0, "Frame should not be empty"