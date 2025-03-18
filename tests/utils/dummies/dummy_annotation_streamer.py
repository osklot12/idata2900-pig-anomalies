from typing import Callable, Optional

import time
from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer
from tests.utils.test_annotation_class import TestBehaviorClass


class DummyAnnotationStreamer(AnnotationStreamer):
    """A testable implementation of the abstract class AnnotationStreamer."""

    def __init__(self, n_annotations: int, callback: Callable[[FrameAnnotation], FeedStatus]):
        """
        Initializes a DummyAnnotationStreamer.

        Args:
            n_annotations (int): number of annotations in the stream
        """
        super().__init__(callback)
        self.n_annotations = n_annotations
        self.callback = callback
        self.frame_index = 0

    def _get_next_annotation(self) -> Optional[FrameAnnotation]:
        annotation = None
        print(f"this ran once")

        if self.frame_index < self.n_annotations:
            time.sleep(.005)
            annotations_list = [
                (TestBehaviorClass.CODING, 3242.234, 3432.4, 234.4, 895.4)
            ]
            annotation = FrameAnnotation("test-source", self.frame_index, annotations_list, False)

            self.frame_index += 1
            print(f"ran this shit")

        return annotation