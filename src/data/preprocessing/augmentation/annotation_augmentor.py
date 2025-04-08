from math import radians, cos, sin
from typing import List

import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.preprocessing.augmentation.augmentor import Augmentor
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory


class AnnotationAugmentor(Augmentor[List[AnnotatedBBox]]):
    """Augments annotations, assuming normalized bounding boxes in the range [0, 1]"""

    def __init__(self, plan_factory: AugmentationPlanFactory):
        """
        Initializes an AnnotationAugmentor instance.

        Args:
            plan_factory (AugmentationPlanFactory): factory for creating augmentation plans
        """
        self._plan_factory = plan_factory

    def augment(self, instance: List[AnnotatedBBox]) -> List[List[AnnotatedBBox]]:
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

    @staticmethod
    def _rotate_annotations(boxes: List[AnnotatedBBox], angle: float) -> List[AnnotatedBBox]:
        """Rotates bounding boxes around the center of the image."""
        cx, cy = 0.5, 0.5
        theta = radians(angle)
        rotation_matrix = np.array([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)]
        ])

        def rotate_bbox(bbox: BBox) -> BBox:
            corners = np.array([
                [bbox.x, bbox.y],
                [bbox.x + bbox.width, bbox.y],
                [bbox.x, bbox.y + bbox.height],
                [bbox.x + bbox.width, bbox.y + bbox.height]
            ])
            rotated = (rotation_matrix @ (corners - [cx, cy]).T).T + [cx, cy]
            x_min, y_min = rotated.min(axis=0)
            x_max, y_max = rotated.max(axis=0)
            return BBox(x=x_min, y=y_min, width=x_max - x_min, height=y_max - y_min)

        return [
            AnnotatedBBox(cls=a.cls, bbox=rotate_bbox(a.bbox))
            for a in boxes
        ]
