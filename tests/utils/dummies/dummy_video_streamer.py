from typing import Optional, Callable

import numpy as np

from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.frame_resize_strategy import FrameResizeStrategy
from src.data.streaming.streamers.video_streamer import VideoStreamer


class DummyVideoStreamer(VideoStreamer):
    """A testable implementation of the abstract class VideoStreamer."""

    def __init__(self, n_frames: int, callback: Callable[[Frame], FeedStatus],
                 resize_strategy: FrameResizeStrategy = None):
        """
        Initializes a DummyVideoStreamer instance.

        Args:
            n_frames (int): the number of frames in the stream
            callback (Callable[[Frame], FeedStatus]): callback function that is fed with frames
            resize_strategy (FrameResizeStrategy): the frame resize strategy
        """
        super().__init__(callback, resize_strategy)
        self.n_frames = n_frames
        self.frame_index = 0
        self.source = "test-source"

    def _get_next_frame(self) -> Optional[Frame]:
        frame = None

        if self.frame_index < self.n_frames:
            frame_data = np.random.randint(0, 256, (1920, 1080, 3), dtype=np.uint8)
            frame = Frame(self.source, self.frame_index, frame_data, False)
            self.frame_index += 1

        return frame