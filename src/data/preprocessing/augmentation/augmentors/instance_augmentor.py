from typing import List, Tuple

import cv2
import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.preprocessing.augmentation.augmentors.augmentor import Augmentor, T
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory
from src.data.preprocessing.augmentation.transformator import Transformator


class InstanceAugmentor(Augmentor[StreamedAnnotatedFrame]):
    """Augments frames."""

    def __init__(self, plan_factory: AugmentationPlanFactory):
        """
        Initializes a FrameAugmentor instance.

        Args:
            plan_factory (AugmentationPlanFactory): factory for creating augmentation plans
        """
        self._plan_factory = plan_factory

    def augment(self, data: StreamedAnnotatedFrame) -> StreamedAnnotatedFrame:
        frame_width, frame_height = data.frame.data.shape[1], data.frame.data.shape[0]
        transform = self._plan_factory.get_current_plan().transform
        shifted_transform = Transformator.compute_shift_and_transform_matrix(
            t=transform, cx=frame_width / 2, cy=frame_height / 2
        )

        return StreamedAnnotatedFrame(
            source=data.source,
            index=data.index,
            frame=self._augment_frame(
                data=data.frame,
                transform=shifted_transform
            ),
            annotations=self._augment_annotations(
                annotations=data.annotations,
                transform=shifted_transform,
                frame_shape=(frame_width, frame_height)
            )
        )

    @staticmethod
    def _augment_frame(data: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Augments an image using the given transformation matrix."""
        width, height = data.shape[1], data.shape[0]
        affine_2x3 = transform[:2, :]

        return cv2.warpAffine(
            data,
            affine_2x3,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

    @staticmethod
    def _augment_annotations(annotations: List[AnnotatedBBox], transform: np.ndarray,
                             frame_shape: Tuple[int, int]) -> List[AnnotatedBBox]:
        """Augments annotations using given transformation matrix."""
        augmented = []

        for anno in annotations:
            bbox = anno.bbox
            absolute_bbox = InstanceAugmentor._get_absolute_bbox(bbox, frame_shape)
            corners = InstanceAugmentor._get_corner_points(absolute_bbox)

            transformed = [transform @ corner for corner in corners]
            xs = [pt[0] for pt in transformed]
            ys = [pt[1] for pt in transformed]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)

            width = x_max - x_min
            height = y_max - y_min

            augmented.append(AnnotatedBBox(
                cls=anno.cls,
                bbox=InstanceAugmentor._get_normalized_bbox(
                    BBox(x=x_min, y=y_min, width=width, height=height),
                    frame_shape
                )
            ))

        return augmented

    @staticmethod
    def _get_corner_points(bbox: BBox) -> List[np.ndarray]:
        """Extracts the corner point coordinates from the bounding box."""
        return [
            np.array([bbox.x, bbox.y, 1]),
            np.array([bbox.x + bbox.width, bbox.y, 1]),
            np.array([bbox.x, bbox.y + bbox.height, 1]),
            np.array([bbox.x + bbox.width, bbox.y + bbox.height, 1])
        ]

    @staticmethod
    def _get_absolute_bbox(bbox: BBox, frame_shape: Tuple[float, float]) -> BBox:
        """Converts a normalized bounding box to absolute pixel values."""
        width, height = frame_shape
        return BBox(
            x=bbox.x * width,
            y=bbox.y * height,
            width=bbox.width * width,
            height=bbox.height * height
        )

    @staticmethod
    def _get_normalized_bbox(bbox: BBox, frame_shape: Tuple[float, float]) -> BBox:
        """Converts an absolute bounding box to normalized values."""
        width, height = frame_shape
        return BBox(
            x=bbox.x / width,
            y=bbox.y / height,
            width=bbox.width / width,
            height=bbox.height / height
        )
