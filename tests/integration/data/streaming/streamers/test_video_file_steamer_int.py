from unittest.mock import MagicMock

import pytest

from src.data.dataclasses.frame import Frame
from src.data.dataset.entities.lazy_video_file import LazyVideoFile
from src.data.preprocessing.resizing.resizers.static_frame_resizer import StaticFrameResizer
from src.data.streaming.streamers.video_file_streamer import VideoFileStreamer
from src.utils.path_finder import PathFinder
from tests.utils.dummies.dummy_frame_resize_strategy import DummyFrameResizeStrategy
from tests.utils.dummies.dummy_video_loader import DummyVideoLoader


@pytest.fixture
def resize_strategy():
    """Fixture to provide a FrameResizeStrategy instance."""
    return StaticFrameResizer((640, 640))

@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return MagicMock()

@pytest.fixture
def dummy_video():
    """Fixture to provide a test video object."""
    video_path = PathFinder.get_abs_path("tests/data/sample-5s.mp4")
    return LazyVideoFile(str(video_path), DummyVideoLoader())

@pytest.fixture
def expected_frames():
    """Fixture to provide the expected number of frames."""
    return 171

@pytest.mark.integration
def test_video_file_streamer_streams_all_frames_without_resizer(dummy_video, mock_callback, expected_frames):
    """Tests that VideoFileStreamer correctly streams all frames from a video file without resizing."""
    # arrange
    streamer = VideoFileStreamer(dummy_video, mock_callback)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count == expected_frames

    for call_args in mock_callback.call_args_list:
        frame = call_args[0][0]
        assert isinstance(frame, Frame)

@pytest.mark.integration
def test_video_file_streamer_streams_all_frames_with_resizer(dummy_video, mock_callback, resize_strategy, expected_frames):
    """Tests that VideoFileStreamer correctly streams all frames from a video file with resizing."""
    # arrange
    streamer = VideoFileStreamer(dummy_video, mock_callback, resize_strategy)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count == expected_frames

    for call_args in mock_callback.call_args_list:
        frame = call_args[0][0]
        assert isinstance(frame, Frame)