from typing import List
import random

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from tests.utils.dummy_annotation_label import DummyAnnotationLabel


class DummyAnnotationsGenerator:
    """A generator of dummy annotations."""

    @staticmethod
    def generate(n: int) -> List[AnnotatedBBox]:
        """
        Generates dummy annotations.

        Args:
            n (int): the number of dummy annotations

        Returns:
            List[AnnotatedBBox]: a list of annotated bboxes
        """
        return [
            AnnotatedBBox(
                cls=random.choice(list(DummyAnnotationLabel)),
                bbox=BBox(
                    x=random.random(),
                    y=random.random(),
                    width=random.random(),
                    height=random.random()
                )
            )
            for _ in range(n)
        ]