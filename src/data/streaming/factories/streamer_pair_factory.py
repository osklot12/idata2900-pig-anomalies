from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from src.data.streaming.streamers.video_streamer import VideoStreamer


class StreamerPairFactory(ABC):
    """A factory for creating pairs of video and annotation streamers."""

    @abstractmethod
    def create_streamer_pair(self, frame_cb: Callable[[Frame], None],
                             annotation_cb: Callable[[FrameAnnotations], None]
                             ) -> Optional[Tuple[VideoStreamer, AnnotationStreamer]]:
        """
        Creates a pair of video and annotation streamers.

        Args:
            frame_cb (Callable[[Frame], FeedStatus]): the callback to feed video frames
            annotation_cb (Callable[[FrameAnnotation], FeedStatus]): the callback to feed annotations

        Returns:
            Tuple[VideoStreamer, AnnotationStreamer]: the pair of video and annotation streamers, or None is not available
        """
        raise NotImplementedError