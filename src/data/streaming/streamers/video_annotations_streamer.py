import queue
from typing import Optional, Callable

from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.normalization.bbox_normalization_strategy import BBoxNormalizationStrategy
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer


class VideoAnnotationsStreamer(AnnotationStreamer):
    """A streamer for streaming video annotations data."""

    def __init__(self, annotations: VideoAnnotations, callback: Callable[[FrameAnnotation], FeedStatus],
                 bbox_normalizer: BBoxNormalizationStrategy):
        """
        Initializes an VideoAnnotationStreamer instance.

        Args:
            annotations (VideoAnnotations): the video annotation data
            callback (Callable[[Annotation], FeedStatus]): the callback function to feed data
            bbox_normalizer (BBoxNormalizationStrategy): the bounding box normalization strategy
        """
        super().__init__(callback, bbox_normalizer)
        self._annotations = annotations
        self._frame_queue = queue.Queue()

    def _setup_stream(self) -> None:
        frame_annotations = self._annotations.get_data()

        for frame_annotation in frame_annotations:
            self._frame_queue.put(frame_annotation)

    def _get_next_annotation(self) -> Optional[FrameAnnotation]:
        return self._frame_queue.get() if not self._frame_queue.empty() else None