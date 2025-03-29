import random
import numpy as np

from src.data.preprocessing.augmentation.annotation_augmentor import AnnotationAugmentor
from src.data.preprocessing.augmentation.augmentor_interface import AugmentorBase
from src.data.preprocessing.augmentation.image_augmentor import ImageAugmentor


class CombinedAugmentation(AugmentorBase):
    """Applies multiple augmentation passes per frame based on config."""

    def __init__(self, rotation_range=None, num_versions=1):
        self.image_augmentor = ImageAugmentor()
        self.annotation_augmentor = AnnotationAugmentor()

        # Default rotation range if not provided
        self.rotation_range = rotation_range if rotation_range else [-15, 15]

        # Number of augmented versions per frame (default 1)
        self.num_versions = max(1, num_versions)

    def augment(self, image: np.ndarray, annotation_list: list, rotation: float = 0, flip: bool = False):
        """Applies augmentation multiple times based on num_versions."""
        augmented_data = []

        for _ in range(self.num_versions):
            flip = random.choice([True, False])
            rotation = random.uniform(self.rotation_range[0], self.rotation_range[1])

            # Apply image augmentation
            aug_image = self.image_augmentor.augment(image, rotation=rotation, flip=flip)

            # Apply annotation augmentation separately
            _, _, aug_annotations = self.annotation_augmentor.augment(
                "source", 0, annotation_list, image.shape[:2], flip=flip, rotation=rotation
            )

            augmented_data.append((aug_image, aug_annotations))

        return augmented_data  # List of (image, annotations) pairs
