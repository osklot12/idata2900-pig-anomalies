from abc import ABC, abstractmethod

from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus


class Aggregator(ABC):
    """An interface for classes that aggregate frames and annotations."""

    @abstractmethod
    def feed_frame(self, frame: Frame) -> None:
        """
        Feeds a frame to the aggregator.

        Args:
            frame (Frame): The frame to feed.
        """
        raise NotImplementedError

    @abstractmethod
    def feed_annotations(self, annotations: FrameAnnotations) -> None:
        """
        Feeds an annotation to the aggregator.

        Args:
            annotations (FrameAnnotations): The annotation the feed.
        """
        raise NotImplementedError