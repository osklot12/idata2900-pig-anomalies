from unittest.mock import MagicMock

import pytest

from src.data.dataclasses.frame import Frame
from src.data.streaming.streamers.streamer_status import StreamerStatus
from tests.utils.dummies.dummy_frame_resize_strategy import DummyFrameResizeStrategy
from tests.utils.dummies.dummy_video_streamer import DummyVideoStreamer


@pytest.fixture
def dummy_resize_strategy():
    """Fixture to provide a dummy resize strategy."""
    return DummyFrameResizeStrategy((720, 1280))


@pytest.fixture
def mock_callback():
    """Fixture to provide a mock callback."""
    return MagicMock()


@pytest.mark.unit
@pytest.mark.parametrize("n_frames", [0, 1, 5, 10])
def test_video_streamer_produces_and_feeds_expected_number_of_frames(n_frames, mock_callback):
    """Tests that the VideoStreamer produces and feeds the correct number of frames."""
    # arrange
    streamer = DummyVideoStreamer(n_frames, mock_callback)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert mock_callback.call_count == n_frames

    for call_args in mock_callback.call_args_list:
        frame = call_args[0][0]
        assert isinstance(frame, Frame)

    assert streamer.get_status() == StreamerStatus.COMPLETED


@pytest.mark.unit
def test_video_streamer_resizes_all_frames(mock_callback, dummy_resize_strategy):
    """Tests that the VideoStreamer resizes all frames while streaming."""
    # arrange
    streamer = DummyVideoStreamer(5, mock_callback, dummy_resize_strategy)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    for call_args in mock_callback.call_args_list:
        frame = call_args[0][0]
        assert isinstance(frame, Frame)
        assert frame.data.shape == (*dummy_resize_strategy.resize_shape, 3)


@pytest.mark.unit
def test_video_streamer_should_indicate_stopping_when_stopped_early(mock_callback):
    """Tests that the VideoStreamer indicates that it was stopped when stopped early."""
    # arrange
    streamer = DummyVideoStreamer(100, mock_callback)
    streamer.start_streaming()

    # act
    streamer.stop_streaming()

    # assert
    assert streamer.get_status() == StreamerStatus.STOPPED
