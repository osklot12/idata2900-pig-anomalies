from typing import Optional, List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.pipeline.consumer import Consumer
from src.data.pipeline.producer import Producer
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.structures.atomic_var import AtomicVar


class BBoxNormalizerComponent(Consumer[FrameAnnotations], Producer[FrameAnnotations]):
    """Pipeline component adapter for normalizing bounding boxes."""

    def __init__(self, normalizer: BBoxNormalizer, consumer: Optional[Consumer[FrameAnnotations]] = None):
        """
        Initializes a BBoxNormalizerComponent.

        Args:
            normalizer (BBoxNormalizer): the normalizer to use
            consumer (Optional[Consumer[FrameAnnotations]]): optional consumer to receive normalized annotations
        """
        self._normalizer = normalizer
        self._consumer = AtomicVar[Consumer[FrameAnnotations]](consumer)

    def consume(self, data: Optional[FrameAnnotations]) -> bool:
        normalized = self._normalize_frame_annotations(data)
        return self._consumer.get().consume(normalized)

    def set_consumer(self, consumer: Consumer[FrameAnnotations]) -> None:
        self._consumer.set(consumer)

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