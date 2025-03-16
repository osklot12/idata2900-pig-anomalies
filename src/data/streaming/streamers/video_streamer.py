from abc import abstractmethod
from typing import Callable, Optional

from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.frame_resize_strategy import FrameResizeStrategy
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class VideoStreamer(Streamer):
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
        frame = self._get_next_frame()
        while frame is not None and not self._is_requested_to_stop():
            if self._resize_strategy:
                frame_data = self._resize_strategy.resize_frame(frame.data)
                frame = Frame(frame.source, frame.index, frame_data, frame.end_of_stream)

            self._callback(frame)
            frame = self._get_next_frame()

        return StreamerStatus.STOPPED if frame is None and self._is_requested_to_stop() else StreamerStatus.COMPLETED

    @abstractmethod
    def _get_next_frame(self) -> Optional[Frame]:
        """
        Returns the next frame for the stream.

        Returns:
            Frame: next Frame, or None if end of stream
        """
        raise NotImplementedError