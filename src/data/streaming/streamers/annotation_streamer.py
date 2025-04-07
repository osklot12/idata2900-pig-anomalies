from abc import abstractmethod

from typing import Optional, List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.streaming.feedables.feedable import Feedable
from src.data.streaming.streamers.concurrent_streamer import ConcurrentStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus


class AnnotationStreamer(ConcurrentStreamer):
    """A streamer for streaming annotation data."""

    def __init__(self, consumer: Feedable[FrameAnnotations], normalizer: Optional[BBoxNormalizer]):
        """
        Initializes a AnnotationStreamer instance.

        Args:
            consumer (Feedable[FrameAnnotations]): the consumer of the streaming data
            normalizer (BBoxNormalizer): the bounding box normalization strategy
        """
        super().__init__()
        self._consumer = consumer
        self._normalizer = normalizer

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.COMPLETED
        annotation = self._get_next_annotation()
        while annotation is not None and not self._is_requested_to_stop():
            normalized_annotation = self._normalize_frame_annotations(annotation)
            self._consumer.feed(normalized_annotation)

            annotation = self._get_next_annotation()

        # indicating end of streams
        self._consumer.feed(None)

        if annotation and self._is_requested_to_stop():
            result = StreamerStatus.STOPPED

        return result

    def _normalize_frame_annotations(self, annotations: FrameAnnotations) -> FrameAnnotations:
        """Normalizes annotations for a frame."""
        if self._normalizer:
            normalized_bboxes = self._get_normalized_bboxes(
                annotations.annotations,
                annotations.source.frame_resolution
            )

            annotations = FrameAnnotations(
                source=annotations.source,
                index=annotations.index,
                annotations=normalized_bboxes
            )

        return annotations

    def _get_normalized_bboxes(self, bboxes: List[AnnotatedBBox], resolution: Tuple[int, int]) -> List[AnnotatedBBox]:
        """Normalizes a list of AnnotatedBBox instances."""
        normalized_bboxes = []

        for anno_box in bboxes:
            normalized_bboxes.append(
                AnnotatedBBox(
                    cls=anno_box.cls,
                    bbox=self._normalizer.normalize_bounding_box(anno_box.bbox, resolution)
                )
            )

        return normalized_bboxes

    @abstractmethod
    def _get_next_annotation(self) -> Optional[FrameAnnotations]:
        """
        Returns the next annotation for the streams.

        Returns:
            Optional[FrameAnnotations]: the next annotation, or None if end of streams
        """
        raise NotImplementedError