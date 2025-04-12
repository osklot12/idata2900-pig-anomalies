import numpy as np

from src.data.preprocessing.augmentation.augmentors.photometric.photometric_filter import PhotometricFilter
from src.data.structures.runtime_param import RuntimeParam


class GaussianNoiseFilter(PhotometricFilter):
    """Applies gaussian noise to images."""

    def __init__(self, std: RuntimeParam[float]):
        """
        Initializes a GaussianNoiseFilter instance.

        Args:
            std (RuntimeParam[float]): standard deviation of the gaussian noise
        """
        self._std = std

    def apply(self, image: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, self._std.resolve(), image.shape).astype(np.float32)
        noisy_frame = image.astype(np.float32) + noise
        return np.clip(noisy_frame, 0, 255).astype(np.uint8)