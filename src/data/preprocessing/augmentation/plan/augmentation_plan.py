from dataclasses import dataclass
import random

import numpy as np

from src.data.preprocessing.augmentation.transformator import Transformator


@dataclass(frozen=True)
class AugmentationPlan:
    """
    Plan for data augmentation.

    Attributes:
        transform (np.ndarray): the transformation matrix
    """
    transform: np.ndarray

    @staticmethod
    def random() -> "AugmentationPlan":
        """
        Returns a random AugmentationPlan.

        Returns:
            AugmentationPlan: a random AugmentationPlan
        """
        flip_horizontal = random.random() < 0.5
        rotate_degrees = random.uniform(-10, 10)
        dilation = random.uniform(1.0, 1.1)

        transform = np.eye(3)

        if flip_horizontal:
            transform = Transformator.compute_reflection_matrix() @ transform

        transform = Transformator.compute_rotation_matrix(rotate_degrees) @ transform

        transform = Transformator.compute_dilation_matrix(dilation) @ transform

        return AugmentationPlan(transform=transform)