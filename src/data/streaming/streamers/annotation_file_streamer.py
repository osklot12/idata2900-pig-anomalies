from typing import Optional, Callable

from src.data.dataclasses.frame_annotation import FrameAnnotation
from src.data.dataset.entities.lazy_annotation import LazyAnnotationJson
from src.data.loading.feed_status import FeedStatus
from src.data.streaming.streamers.annotation_streamer import AnnotationStreamer


class AnnotationFileStreamer(AnnotationStreamer):
    """A streamer for streaming annotation file data."""

    def __init__(self, json: LazyAnnotationJson, callback: Callable[[FrameAnnotation], FeedStatus]):
        """
        Initializes an AnnotationFileStreamer instance.

        Args:
            json (LazyAnnotationJson): the annotation json
            callback (Callable[[Annotation], FeedStatus]): the callback function to feed data
        """
        super().__init__(callback)
        self._json = json
        self._data = None

    def _setup_stream(self) -> None:
        self._data = self._json.get_data()

    def _get_next_annotation(self) -> Optional[FrameAnnotation]:
        annotation = None

