from typing import Callable, Optional

import time

from src.data.dataclasses.bbox_annotation import BBoxAnnotation
from src.data.dataclasses.bounding_box import BoundingBox
from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.normalization.normalizers.bbox_normalization_strategy import BBoxNormalizationStrategy
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from tests.utils.test_annotation_class import TestAnnotationLabel


class DummyAnnotationStreamer(AnnotationStreamer):
    """A testable implementation of the abstract class AnnotationStreamer."""

    def __init__(self, n_annotations: int, callback: Callable[[FrameAnnotation], FeedStatus],
                 normalizer: BBoxNormalizationStrategy = None):
        """
        Initializes a DummyAnnotationStreamer.

        Args:
            n_annotations (int): number of annotations in the stream
            callback (Callable[[FrameAnnotation], FeedStatus]): the callback to feed streaming annotations
            normalizer (BBoxNormalizationStrategy): the normalization strategy for normalizing bounding boxes
        """
        super().__init__(callback, normalizer)
        self.n_annotations = n_annotations
        self.callback = callback
        self.frame_index = 0

    def _get_next_annotation(self) -> Optional[FrameAnnotation]:
        annotation = None

        if self.frame_index < self.n_annotations:
            time.sleep(.005)
            annotations_list = [
                BBoxAnnotation(
                    cls=TestAnnotationLabel.CODING,
                    bbox=BoundingBox(
                        center_x=1240.6,
                        center_y=980.04,
                        width=234.4,
                        height=895.4
                    )
                )
            ]
            annotation = FrameAnnotation("test-source", self.frame_index, annotations_list, False)

            self.frame_index += 1

        return annotation