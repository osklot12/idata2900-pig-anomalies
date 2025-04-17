from typing import List, Tuple

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.preprocessing.preprocessor import Preprocessor
from src.data.processing.augmentation.augmentation_plan_factory import AugmentationPlanFactory


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