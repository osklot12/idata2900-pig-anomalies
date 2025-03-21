from abc import abstractmethod
from typing import Callable, Optional

from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.resizing.frame_resize_strategy import FrameResizeStrategy
from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class VideoStreamer(ThreadedStreamer):
    """A streamer for streaming video data."""

    def __init__(self, callback: Callable[[Frame], FeedStatus], resize_strategy: Optional[FrameResizeStrategy]):
        """
        Initializes a VideoStreamer instance.

        Args:
            callback (Callable[[str, int, np.ndarray, bool], FeedStatus]): the callback to feed data
            resize_strategy (Optional[FrameResizeStrategy]): the frame resize strategy to use
        """
        super().__init__()
        self._callback = callback
        self._resize_strategy = resize_strategy

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.COMPLETED

        frame = self._get_next_frame()
        while frame is not None and not self._is_requested_to_stop():
            if self._resize_strategy:
                frame_data = self._resize_strategy.resize_frame(frame.data)
                frame = Frame(frame.source, frame.index, frame_data, frame.end_of_stream)

            self._callback(frame)
            frame = self._get_next_frame()

        if frame and self._is_requested_to_stop():
            result = StreamerStatus.STOPPED

        return result

    @abstractmethod
    def _get_next_frame(self) -> Optional[Frame]:
        """
        Returns the next frame for the stream.

        Returns:
            Optional[Frame]: next Frame, or None if end of stream
        """
        raise NotImplementedError