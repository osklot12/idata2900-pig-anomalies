from dataclasses import dataclass
import random

@dataclass(frozen=True)
class AugmentationPlan:
    """
    Plan for data augmentation.

    Attributes:
        flip_horizontal (bool): whether to apply horizontal flip
        rotate_degrees (float): rotation angle in degrees (counterclockwise)
    """
    flip_horizontal: bool
    rotate_degrees: float

    @staticmethod
    def random() -> "AugmentationPlan":
        """
        Returns a random AugmentationPlan.

        Returns:
            AugmentationPlan: a random AugmentationPlan
        """
        return AugmentationPlan(
            flip_horizontal=random.random() < 0.5,
            rotate_degrees=random.uniform(-15, 15)
        )