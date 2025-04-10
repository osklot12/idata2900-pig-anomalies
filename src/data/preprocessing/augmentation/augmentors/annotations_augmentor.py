from typing import List, Tuple

import cv2
import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.preprocessing.preprocessor import Preprocessor
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory


class AnnotationsAugmentor(Preprocessor[List[AnnotatedBBox]]):
    """Augments annotations, assuming normalized bounding boxes in the range [0, 1]"""

    def __init__(self, plan_factory: AugmentationPlanFactory, image_shape: Tuple[int, int]):
        """
        Initializes an AnnotationAugmentor instance.

        Args:
            plan_factory (AugmentationPlanFactory): factory for creating augmentation plans
            image_shape (Tuple[int, int]): shape of the image (width, height)
        """
        self._plan_factory = plan_factory
        self._image_shape = image_shape

    def process(self, instance: List[AnnotatedBBox]) -> List[List[AnnotatedBBox]]:
        plan = self._plan_factory.get_current_plan()

        if plan.flip_horizontal:
            instance = self._flip_annotations(instance)

        if plan.rotate_degrees != 0:
            instance = self._rotate_annotations(instance, plan.rotate_degrees)

        return [instance]

    @staticmethod
    def _flip_annotations(boxes: List[AnnotatedBBox]) -> List[AnnotatedBBox]:
        """Flips bounding boxes horizontally."""
        return [
            AnnotatedBBox(
                cls=a.cls,
                bbox=BBox(
                    x=1 - a.bbox.x - a.bbox.width,
                    y=a.bbox.y,
                    width=a.bbox.width,
                    height=a.bbox.height
                )
            )
            for a in boxes
        ]

    def _rotate_annotations(self, boxes: List[AnnotatedBBox], angle: float) -> List[AnnotatedBBox]:
        """Rotates bounding boxes around the center of the image."""
        frame_width, frame_height = self._image_shape
        center = (frame_width / 2, frame_height / 2)

        # Get OpenCV rotation matrix (2x3)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        rotated_boxes = []

        for ann in boxes:
            # Convert bbox to pixel coordinates
            x = ann.bbox.x * frame_width
            y = ann.bbox.y * frame_height
            w = ann.bbox.width * frame_width
            h = ann.bbox.height * frame_height

            # Get the 4 corners of the box
            corners = np.array([
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h]
            ], dtype=np.float32)

            # Apply affine transform to the corners
            rotated_corners = cv2.transform(np.array([corners]), rot_matrix)[0]

            # Get new axis-aligned bounding box in pixel space
            x_min, y_min = rotated_corners.min(axis=0)
            x_max, y_max = rotated_corners.max(axis=0)

            # Convert back to normalized coordinates
            new_bbox = BBox(
                x=x_min / frame_width,
                y=y_min / frame_height,
                width=(x_max - x_min) / frame_width,
                height=(y_max - y_min) / frame_height
            )

            # Clamp to valid range
            new_bbox = BBox(
                x=max(0.0, min(1.0, new_bbox.x)),
                y=max(0.0, min(1.0, new_bbox.y)),
                width=max(0.0, min(1.0 - new_bbox.x, new_bbox.width)),
                height=max(0.0, min(1.0 - new_bbox.y, new_bbox.height))
            )

            rotated_boxes.append(AnnotatedBBox(cls=ann.cls, bbox=new_bbox))

        return rotated_boxes
