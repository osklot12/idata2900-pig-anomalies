from abc import ABC, abstractmethod
from typing import Optional

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus


class Aggregator(ABC):
    """An interface for classes that aggregate frames and annotations."""

    @abstractmethod
    def feed_frame(self, frame: Optional[Frame]) -> None:
        """
        Feeds a frame to the aggregator.

        Args:
            frame (Frame): the frame to feed, or None if end of stream
        """
        raise NotImplementedError

    @abstractmethod
    def feed_annotations(self, annotations: Optional[FrameAnnotations]) -> None:
        """
        Feeds an annotation to the aggregator.

        Args:
            annotations (FrameAnnotations): the annotations to feed, or None if end of stream
        """
        raise NotImplementedError