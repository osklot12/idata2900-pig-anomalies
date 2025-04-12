import cv2
import numpy as np

from src.data.preprocessing.augmentation.augmentors.photometric.photometric_filter import PhotometricFilter
from src.data.structures.runtime_param import RuntimeParam


class ColorJitterFilter(PhotometricFilter):
    """Applies color jitter to images."""

    def __init__(self, saturation_scale: RuntimeParam[float], hue_shift: RuntimeParam[float]):
        """
        Initializes a ColorJitterFilter instance.

        Args:
            saturation_scale (RuntimeParam[float]): saturation scale value
            hue_shift (RuntimeParam[float]): hue shift value
        """
        self._saturation = saturation_scale
        self._hue_shift = hue_shift

    def apply(self, image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= self._saturation.resolve()
        hsv[..., 0] += self._hue_shift.resolve()
        hsv[..., 0] %= 180
        hsv = np.clip(hsv, 0, 255)

        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)