from abc import abstractmethod

from typing import Callable, Optional

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.loading.feed_status import FeedStatus
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.streaming.streamers.threaded_streamer import ThreadedStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class AnnotationStreamer(ThreadedStreamer):
    """A streamer for streaming annotation data."""

    def __init__(self, callback: Callable[[FrameAnnotations], FeedStatus], normalizer: Optional[BBoxNormalizer]):
        """
        Initializes a AnnotationStreamer instance.

        Args:
            callback (Callable[[Annotation], FeedStatus]): the callback to feed data
            normalizer (BBoxNormalizer): the bounding box normalization strategy
        """
        super().__init__()
        self._callback = callback
        self._normalizer = normalizer

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.COMPLETED

        annotation = self._get_next_annotation()
        while annotation is not None and not self._is_requested_to_stop():
            normalized_annotation = self._normalize_annotation(annotation)
            self._callback(normalized_annotation)

            annotation = self._get_next_annotation()

        if annotation and self._is_requested_to_stop():
            result = StreamerStatus.STOPPED

        return result

    def _normalize_annotation(self, annotation: FrameAnnotations) -> FrameAnnotations:
        """Normalizes an annotation."""
        if self._normalizer:
            normalized_bboxes = []
            for anno_box in annotation.annotations:
                normalized_bboxes.append(
                    AnnotatedBBox(
                        cls=anno_box.cls,
                        bbox=self._normalizer.normalize_bounding_box(anno_box.bbox)
                    )
                )

            annotation = FrameAnnotations(
                source=annotation.source,
                index=annotation.index,
                annotations=normalized_bboxes,
                end_of_stream=annotation.end_of_stream,
            )

        return annotation

    @abstractmethod
    def _get_next_annotation(self) -> Optional[FrameAnnotations]:
        """
        Returns the next annotation for the stream.

        Returns:
            Optional[FrameAnnotations]: the next annotation, or None if end of stream
        """
        raise NotImplementedError