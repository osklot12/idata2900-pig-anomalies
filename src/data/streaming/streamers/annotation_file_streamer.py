from typing import Optional, Callable

from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.dataset.entities.lazy_video_annotations import LazyVideoAnnotations
from src.data.dataset.entities.video_annotations import VideoAnnotations
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer


class AnnotationFileStreamer(AnnotationStreamer):
    """A streamer for streaming annotation file data."""

    def __init__(self, annotations: VideoAnnotations, callback: Callable[[FrameAnnotation], FeedStatus]):
        """
        Initializes an AnnotationFileStreamer instance.

        Args:
            annotations (VideoAnnotations): the video annotation data
            callback (Callable[[Annotation], FeedStatus]): the callback function to feed data
        """
        super().__init__(callback)
        self._annotations = annotations
        self._data = None

    def _setup_stream(self) -> None:
        self._data = self._annotations.get_data()

    def _get_next_annotation(self) -> Optional[FrameAnnotation]:
