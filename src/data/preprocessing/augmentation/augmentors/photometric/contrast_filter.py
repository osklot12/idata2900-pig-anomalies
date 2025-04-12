import cv2
import numpy as np

from src.data.preprocessing.augmentation.augmentors.photometric.photometric_filter import PhotometricFilter
from src.data.structures.runtime_param import RuntimeParam


class ContrastFilter(PhotometricFilter):
    """Applies contrast to images."""

    def __init__(self, alpha: RuntimeParam[float]):
        """
        Initializes a ContrastFilter instance.

        Args:
            alpha (RuntimeParam[float]): contrast alpha value
        """
        self._alpha = alpha

    def apply(self, image: np.ndarray) -> np.ndarray:
        return cv2.convertScaleAbs(image, alpha=self._alpha.resolve())
