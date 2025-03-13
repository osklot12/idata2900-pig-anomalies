from abc import ABC, abstractmethod
import tensorflow as tf

class AugmentorBase(ABC):
    """Abstract base class for all augmentors."""

    @abstractmethod
    def augment(self, image: tf.Tensor, annotation: dict = None, rotation: float = 0, flip: bool = False):
        """Applies augmentation to the given image and/or annotation."""
        pass


class ProcessorBase(ABC):
    """Abstract base class for all image processing components."""

    @abstractmethod
    def process(self, image: tf.Tensor) -> tf.Tensor:
        """Processes an image tensor."""
        pass