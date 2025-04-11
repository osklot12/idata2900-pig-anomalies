from abc import ABC, abstractmethod
from typing import Tuple, Optional

from src.data.dataclasses.frame import Frame
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from src.data.streaming.streamers.video_streamer import VideoStreamer


class StreamerPairProvider(ABC):
    """A factory for creating pairs of video and annotation streamers."""

    @abstractmethod
    def create_streamer_pair(self, frame_consumer: Consumer[Frame], annotations_consumer: Consumer[FrameAnnotations]
                             ) -> Optional[Tuple[VideoStreamer, AnnotationStreamer]]:
        """
        Creates a pair of video and annotation streamers.

        Args:
            frame_consumer (Consumer[Frame]): consumer of the streamed frames
            annotations_consumer (Feedable[FrameAnnotations]): consumer of the streamed annotations

        Returns:
            Tuple[VideoStreamer, AnnotationStreamer]: the pair of video and annotation streamers, or None is not available
        """
        raise NotImplementedError