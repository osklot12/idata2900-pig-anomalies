import pytest
from cppbindings import FrameStream

from src.data.dataclasses.frame import Frame
from src.utils.path_finder import PathFinder
from tests.utils.dummies.dummy_video_loader import DummyVideoLoader


@pytest.fixture
def video_loader():
    """Fixture to provide a VideoLoader."""
    return DummyVideoLoader()

@pytest.fixture
def video_file(video_loader):
    """Fixture to provide a test video file."""
    path = str(PathFinder.get_abs_path("tests/data/sample-5s.mp4"))
    return bytearray(video_loader.load_video_file(path))

def test_frame_stream_extracts_all_frames(video_file):
    """Tests that FrameStream extracts the expected number of frames for a video."""
    # arrange
    fstream = FrameStream(video_file)
    frames = []

    # act
    frame = fstream.read()
    while frame:
        frames.append(frame)
        frame = fstream.read()

    # assert
    assert len(frames) == 171
    for frame in frames:
        assert frame is not None
