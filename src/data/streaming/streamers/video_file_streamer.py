from typing import Optional, Callable

from cppbindings import FrameStream

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.video import Video
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.frame_resize_strategy import FrameResizeStrategy
from src.data.streaming.streamers.video_streamer import VideoStreamer


class VideoFileStreamer(VideoStreamer):
    """A streamer for streaming video file data."""

    def __init(self, video: Video, callback: Callable[[Frame], FeedStatus],
               resize_strategy: FrameResizeStrategy = None):
        """
        Initializes a VideoFileStreamer instance.

        Args:
            video (Video): the video to stream
            callback (Callable[[Frame], FeedStatus]): the callback function to feed data
            resize_strategy (Optional[FrameResizeStrategy]): the frame resize strategy to use
        """
        super().__init__(callback, resize_strategy)
        self._video = video
        self._fstream = None
        self._frame_index = 0

    def _setup_stream(self) -> None:
        video_data = self._video.loader.load_video(self._video.id)
        self._fstream = FrameStream(video_data)

    def _get_next_frame(self) -> Optional[Frame]:
        frame_data = self._fstream.read()
        end_of_stream = not frame_data
        frame = Frame(self._video.id, self._frame_index, frame_data, end_of_stream)

        self._frame_index += 1

        return frame