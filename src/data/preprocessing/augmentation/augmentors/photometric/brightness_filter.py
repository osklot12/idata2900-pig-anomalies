import cv2
import numpy as np

from src.data.preprocessing.augmentation.augmentors.photometric.photometric_filter import PhotometricFilter
from src.data.structures.runtime_param import RuntimeParam


class BrightnessFilter(PhotometricFilter):
    """Applies brightness to images."""

    def __init__(self, beta: RuntimeParam[float]):
        """
        Initializes a BrightnessFilter instance.

        Args:
            beta (RuntimeParam[float]): brightness beta value
        """
        self._beta = beta

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, beta=self._beta.resolve())