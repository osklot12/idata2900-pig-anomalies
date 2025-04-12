from abc import abstractmethod

from typing import Optional, List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.pipeline.producer import Producer
from src.data.pipeline.consumer import Consumer
from src.data.streaming.streamers.concurrent_streamer import ConcurrentStreamer
from src.data.streaming.streamers.streamer_status import StreamerStatus
from src.data.structures.atomic_var import AtomicVar


class AnnotationStreamer(Producer[FrameAnnotations], ConcurrentStreamer):
    """A streamer for streaming annotation data."""

    def __init__(self, consumer: Optional[Consumer[FrameAnnotations]] = None):
        """
        Initializes a AnnotationStreamer instance.

        Args:
            consumer (Optional[Consumer[FrameAnnotations]]): the consumer of the streamed data
        """
        super().__init__()
        self._consumer = AtomicVar[Consumer[FrameAnnotations]](consumer)

    def _stream(self) -> StreamerStatus:
        result = StreamerStatus.COMPLETED
        annotation = self._get_next_annotation()
        while annotation is not None and not self._is_requested_to_stop():
            consumer = self._consumer.get()
            if consumer is not None:
                consumer.consume(annotation)
                annotation = self._get_next_annotation()

        # indicating end of streams
        consumer = self._consumer.get()
        if consumer is not None:
            consumer.consume(annotation)

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

    def connect(self, consumer: Consumer[FrameAnnotations]) -> None:
        self._consumer.set(consumer)