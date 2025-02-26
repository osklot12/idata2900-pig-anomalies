import random
import numpy as np
from src.data.augment.image_augmentor import ImageAugmentor
from src.data.augment.annotation_augmentor import AnnotationAugmentor

class CombinedAugmentation:
    """Combines image and annotation augmentation into one unified process."""

    def __init__(self, seed=None):
        self.image_augmentor = ImageAugmentor(seed)
        self.annotation_augmentor = AnnotationAugmentor()

    def augment(self, image: np.ndarray, annotations: list):
        """Applies both image & annotation augmentation consistently."""
        flip = random.choice([True, False])
        rotation = random.uniform(-15, 15)

        # Apply image augmentation
        aug_image = self.image_augmentor.augment(image, rotation=rotation, flip=flip)

        # Apply annotation augmentation
        _, _, aug_annotations = self.annotation_augmentor.augment(
            "source", 0, annotations, image.shape[:2], flip=flip, rotation=rotation
        )

        return aug_image, aug_annotations
