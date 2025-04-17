from abc import ABC, abstractmethod

import numpy as np


class PhotometricFilter(ABC):
    """Interface for photometric augmenters."""

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """
        Applies the photometric augmentation.

        Args:
            image (np.ndarray): image to be augmented

        Returns:
            np.ndarray: the augmented image
        """
        raise NotImplementedError