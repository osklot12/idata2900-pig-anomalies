from typing import List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.annotated_frame import AnnotatedFrame
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer
from src.data.processing.processor import Processor


class BBoxNormalizerProcessor(Processor[AnnotatedFrame, AnnotatedFrame]):
    """Processor for normalizing bounding boxes."""

    def __init__(self, normalizer: BBoxNormalizer):
        """
        Initializes a BBoxNormalizerProcessor instance.

        Args:
            normalizer (BBoxNormalizer): normalizer to use
        """
        self._normalizer = normalizer

    def process(self, data: AnnotatedFrame) -> AnnotatedFrame:
        return self._normalize_frame_annotations(data)

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
