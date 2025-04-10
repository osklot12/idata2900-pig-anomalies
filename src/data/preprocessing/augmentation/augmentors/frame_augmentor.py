from typing import List

import cv2
import random
import numpy as np

from src.data.preprocessing.preprocessor import Preprocessor
from src.data.preprocessing.augmentation.plan.augmentation_plan_factory import AugmentationPlanFactory


class FrameAugmentor(Preprocessor[np.ndarray]):
    """Augments frames."""

    def __init__(self, plan_factory: AugmentationPlanFactory):
        """
        Initializes a FrameAugmentor instance.

        Args:
            plan_factory (AugmentationPlanFactory): factory for creating augmentation plans
        """
        self._plan_factory = plan_factory

    def process(self, instance: np.ndarray) -> List[np.ndarray]:
        plan = self._plan_factory.get_current_plan()

        if plan.flip_horizontal:
            instance = cv2.flip(instance, 1)

        if plan.rotate_degrees != 0:
            instance = self._rotate(instance, plan.rotate_degrees)

        instance = self._random_brightness_contrast(instance)

        return [instance]

    @staticmethod
    def _rotate(image: np.ndarray, angle) -> np.ndarray:
        """Rotates an image while keeping its size unchanged."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_matrix, (w, h))

    @staticmethod
    def _random_brightness_contrast(image: np.ndarray) -> np.ndarray:
        """Applies random brightness and contrast adjustments."""
        if image.dtype != np.uint8:
            raise ValueError("Expected image dtype uint8")
        
        alpha = random.uniform(0.7, 1.3)  # Contrast factor
        beta = random.uniform(-30, 30)  # Brightness factor
        return np.clip(image * alpha + beta, 0, 255).astype(np.uint8)