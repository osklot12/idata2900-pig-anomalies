from abc import ABC, abstractmethod

from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.dataclasses.frame import Frame
from src.data.loading.feed_status import FeedStatus


class InstanceAggregator(ABC):
    """An interface for classes that aggregate frames and annotations."""

    @abstractmethod
    def feed_frame(self, frame: Frame) -> FeedStatus:
        """
        Feeds a frame to the aggregator.

        Args:
            frame (Frame): The frame to feed.

        Returns:
            a FeedStatus indicating how the frame was received.
        """
        raise NotImplementedError

    def feed_annotation(self, annotation: FrameAnnotation) -> FeedStatus:
        """
        Feeds an annotation to the aggregator.

        Args:
            annotation (FrameAnnotation): The annotation the feed.

        Returns:
            a FeedStatus indicating how the annotation was received.
        """
        raise NotImplementedError