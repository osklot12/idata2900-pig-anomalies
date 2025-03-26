from abc import ABC, abstractmethod
import numpy as np

class AugmentorBase(ABC):
    """Abstract base class for all augmentors."""

    @abstractmethod
    def augment(self, image: np.ndarray, annotation_list: list, rotation: float = 0, flip: bool = False):
        """Applies augmentation to the given image and/or annotation."""
        pass


class ProcessorBase(ABC):
    """Abstract base class for all image processing components."""

    @abstractmethod
    def process(self, image: np.ndarray):
        """Processes an image array."""
        pass
