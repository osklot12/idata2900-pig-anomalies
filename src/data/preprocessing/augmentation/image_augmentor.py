import cv2
import numpy as np
import random

from src.data.preprocessing.augmentation.augmentor_interface import AugmentorBase


class ImageAugmentor(AugmentorBase):
    """
    Applies augmentations like flipping, rotation, brightness, and contrast to images.
    """
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)

    def augment(self, image: np.ndarray, annotation_list=None, rotation: float = 0, flip: bool = False):
        """
        Applies augmentations with specified rotation and flip.

        :param image: Raw RGB image as a NumPy array.
        :param rotation: Rotation angle in degrees.
        :param flip: Whether to apply horizontal flipping.
        :return: Augmented image.
        """
        if flip:
            image = cv2.flip(image, 1)

        if rotation != 0:
            image = self._rotate(image, rotation)

        image = self._random_brightness_contrast(image)
        return image

    def _rotate(self, image, angle):
        """Rotates an image while keeping its size unchanged."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_matrix, (w, h))

    def _random_brightness_contrast(self, image):
        """Applies random brightness and contrast adjustments."""
        alpha = random.uniform(0.7, 1.3)  # Contrast factor
        beta = random.uniform(-30, 30)  # Brightness factor
        return np.clip(image * alpha + beta, 0, 255).astype(np.uint8)
