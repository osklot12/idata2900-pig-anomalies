from abc import abstractmethod
from typing import Callable, Optional

from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.streamers.streamer import Streamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class VideoStreamer(Streamer):
    """A streamer for streaming video data."""

    def __init__(self, callback: Callable[[Frame], FeedStatus]):
        """
        Initializes a VideoStreamer instance.

        Args:
            callback (Callable[[str, int, np.ndarray, bool], FeedStatus]): the callback to feed data
        """
        super().__init__()
        self._callback = callback

    def _stream(self) -> StreamerStatus:
        frame = self._get_next_frame()
        while frame is not None and not self._is_requested_to_stop():

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