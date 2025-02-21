from src.data.augment.image_augmentor import ImageAugmentor
from src.data.augment.json_augmentor import JsonAugmentor
from src.data.augment.augmentor_interface import AugmentorBase
import tensorflow as tf
import random

class AugmentationPipeline(AugmentorBase):
    """Combines image and JSON augmentation into one unified process."""

    def __init__(self, seed=None):
        self.image_augmentor = ImageAugmentor(seed)
        self.json_augmentor = JsonAugmentor()

    def augment(self, image: tf.Tensor, annotation: dict):
        """Applies both image & annotation augmentation consistently."""
        flip = random.choice([True, False])
        rotation = random.uniform(-15, 15)

        aug_image = self.image_augmentor.augment(image, rotation=rotation, flip=flip)
        aug_annotation = self.json_augmentor.augment(image, annotation, rotation=rotation, flip=flip)

        return aug_image, aug_annotation
