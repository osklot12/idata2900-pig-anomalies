from unittest.mock import MagicMock

import pytest

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.video import Video
from src.data.streaming.streamers.video_file_streamer import VideoFileStreamer
from src.utils.path_finder import PathFinder
from tests.utils.dummies.dummy_frame_resize_strategy import DummyFrameResizeStrategy
from tests.utils.dummies.dummy_video_loader import DummyVideoLoader


@pytest.fixture
def dummy_resize_strategy():
    """Fixture to provide a DummyFrameResizeStrategy instance."""
    return DummyFrameResizeStrategy((720, 1280))

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return MagicMock()

@pytest.fixture
def dummy_video():
    """Fixture to provide a test video object."""
    video_path = PathFinder.get_abs_path("tests/data/sample-5s.mp4")
    return Video(id=str(video_path), loader=DummyVideoLoader())

@pytest.mark.integration
def test_video_file_streamer_streams_all_frames(dummy_video, mock_callback, dummy_resize_strategy):
    """Tests that VideoFileStreamer correctly streams all frames from a video file."""
    # arrange
    streamer = VideoFileStreamer(dummy_video, mock_callback, dummy_resize_strategy)
    expected_frames = 171

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count > 0
    assert mock_callback.call_count == expected_frames

    for call_args in mock_callback.call_args_list:
        frame = call_args[0][0]
        assert isinstance(frame, Frame)