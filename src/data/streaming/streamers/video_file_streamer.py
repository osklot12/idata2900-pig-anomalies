from typing import Optional, Callable

from cppbindings import FrameStream

from src.data.dataclasses.frame import Frame
from src.data.dataset.entities.video_file import VideoFile
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.resizing.resizers.frame_resize_strategy import FrameResizeStrategy
from src.data.streaming.streamers.video_streamer import VideoStreamer


class VideoFileStreamer(VideoStreamer):
    """A streamer for streaming video file data."""

    def __init__(self, video: VideoFile, callback: Callable[[Frame], FeedStatus],
                 resize_strategy: FrameResizeStrategy = None):
        """
        Initializes a VideoFileStreamer instance.

        Args:
            video (VideoFile): the video to stream
            callback (Callable[[Frame], FeedStatus]): the callback function to feed data
            resize_strategy (Optional[FrameResizeStrategy]): the frame resize strategy to use
        """
        super().__init__(callback, resize_strategy)
        self._video = video
        self._fstream = None
        self._frame_index = 0

    def _setup_stream(self) -> None:
        video_data = self._video.get_data()
        self._fstream = FrameStream(bytearray(video_data))

    def _get_next_frame(self) -> Optional[Frame]:
        frame = None

        frame_data = self._fstream.read()
        if frame_data:
            frame = Frame(self._video.get_id(), self._frame_index, frame_data, False)
            self._frame_index += 1

        return frame