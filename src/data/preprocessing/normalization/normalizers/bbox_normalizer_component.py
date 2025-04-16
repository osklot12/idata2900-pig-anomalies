from typing import Optional, List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.frame_annotations import FrameAnnotations
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.pipeline.component import Component
from src.data.pipeline.consumer import Consumer
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.structures.atomic_var import AtomicVar


class BBoxNormalizerComponent(Component[AnnotatedFrame, AnnotatedFrame]):
    """Pipeline component adapter for normalizing bounding boxes."""

    def __init__(self, normalizer: BBoxNormalizer, consumer: Optional[Consumer[AnnotatedFrame]] = None):
        """
        Initializes a BBoxNormalizerComponent.

        Args:
            normalizer (BBoxNormalizer): the normalizer to use
            consumer (Optional[Consumer[FrameAnnotations]]): optional consumer to receive normalized annotations
        """
        self._normalizer = normalizer
        self._consumer = AtomicVar[Consumer[AnnotatedFrame]](consumer)

    def consume(self, data: Optional[AnnotatedFrame]) -> bool:
        normalized = None

        if data is not None:
            normalized = self._normalize_frame_annotations(data)

        return self._consumer.get().consume(normalized)

    def connect(self, consumer: Consumer[AnnotatedFrame]) -> None:
        self._consumer.set(consumer)

    def _normalize_frame_annotations(self, instance: AnnotatedFrame) -> AnnotatedFrame:
        """Normalizes annotations for a frame."""
        if self._normalizer:
            normalized_bboxes = self._get_normalized_bboxes(
                instance.annotations,
                instance.source.frame_resolution
            )

            instance = AnnotatedFrame(
                source=instance.source,
                index=instance.index,
                frame=instance.frame,
                annotations=normalized_bboxes
            )

        return instance

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