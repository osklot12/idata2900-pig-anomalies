import time
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.preprocessing.resizing.resizers.frame_resizer import FrameResizer
from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.streamer_status import StreamerStatus
from src.data.streaming.streamers.video_streamer import VideoStreamer
from tests.utils.dummies.dummy_frame_resize_strategy import DummyFrameResizeStrategy


class DummyVideoStreamer(VideoStreamer):
    """A testable implementation of the abstract class VideoStreamer."""

    def __init__(self, n_frames: int, consumer: Consumer[Frame], resizer: FrameResizer = None):
        """
        Initializes a DummyVideoStreamer instance.

        Args:
            n_frames (int): the number of frames in the streams
            consumer (Consumer[Frame]): the consumer that receives the frames
            resizer (FrameResizer): the frame resize strategy
        """
        super().__init__(consumer)
        self.n_frames = n_frames
        self.frame_index = 0
        self.source = SourceMetadata("test-source", (1920, 1080))

    def _get_next_frame(self) -> Optional[Frame]:
        frame = None

        if self.frame_index < self.n_frames:
            time.sleep(.005)
            frame_data = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
            frame = Frame(self.source, self.frame_index, frame_data)
            self.frame_index += 1

        return frame


@pytest.fixture
def dummy_resize_strategy():
    """Fixture to provide a dummy resize strategy."""
    return DummyFrameResizeStrategy((720, 1280))


@pytest.fixture
def consumer():
    """Fixture to provide a mock Feedable."""
    return MagicMock()


@pytest.mark.unit
@pytest.mark.parametrize("n_frames", [0, 1, 5, 10])
def test_video_streamer_produces_and_feeds_expected_number_of_frames(n_frames, consumer):
    """Tests that the VideoStreamer produces and feeds the correct number of frames."""
    # arrange
    streamer = DummyVideoStreamer(n_frames, consumer)

    # act
    streamer.start_streaming()
    streamer.wait_for_completion()

    # assert
    assert consumer.consume.call_count == n_frames + 1

    for i in range(n_frames):
        call_args = consumer.consume.call_args_list[i]
        frame = call_args[0][0]
        assert isinstance(frame, Frame)

    assert consumer.consume.call_args_list[n_frames][0][0] is None

    assert streamer.get_status() == StreamerStatus.COMPLETED


@pytest.mark.unit
def test_video_streamer_should_indicate_stopping_when_stopped_early(consumer):
    """Tests that the VideoStreamer indicates that it was stopped when stopped early."""
    # arrange
    streamer = DummyVideoStreamer(100, consumer)
    streamer.start_streaming()

    # act
    streamer.stop_streaming()

    # assert
    assert streamer.get_status() == StreamerStatus.STOPPED
