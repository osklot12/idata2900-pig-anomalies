from typing import Callable, Optional

import time

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.source_metadata import SourceMetadata
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from tests.utils.annotation_label import AnnotationLabel


class DummyAnnotationStreamer(AnnotationStreamer):
    """A testable implementation of the abstract class AnnotationStreamer."""

    def __init__(self, n_annotations: int, callback: Callable[[FrameAnnotations], FeedStatus],
                 normalizer: BBoxNormalizer = None):
        """
        Initializes a DummyAnnotationStreamer.

        Args:
            n_annotations (int): number of annotations in the stream
            callback (Callable[[FrameAnnotation], FeedStatus]): the callback to feed streaming annotations
            normalizer (BBoxNormalizer): the normalization strategy for normalizing bounding boxes
        """
        super().__init__(callback, normalizer)
        self.n_annotations = n_annotations
        self.callback = callback
        self.frame_index = 0

    def _get_next_annotation(self) -> Optional[FrameAnnotations]:
        annotation = None

        if self.frame_index < self.n_annotations:
            time.sleep(.005)
            annotations_list = [
                AnnotatedBBox(
                    cls=AnnotationLabel.CODING,
                    bbox=BBox(
                        x=1240.6,
                        y=980.04,
                        width=234.4,
                        height=895.4
                    )
                )
            ]
            annotation = FrameAnnotations(
                source=SourceMetadata("test-source-id", (1920, 1080)),
                index=self.frame_index,
                annotations=annotations_list,
                end_of_stream=False
            )

            self.frame_index += 1

        return annotation