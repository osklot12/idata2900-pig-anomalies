import tensorflow as tf
from src.data.preprocessing.augmentation.augmentor_interface import ProcessorBase

class ImageNormalizer(ProcessorBase):
    """Handles normalization of image pixel values."""

    def __init__(self, norm_range=(0, 1)):
        self.norm_range = norm_range
        self._validate_norm_range()

    def process(self, image: tf.Tensor) -> tf.Tensor:
        """Normalizes the image to the specified range."""
        self._validate_input(image)
        return self._normalize(image)

    def _normalize(self, image):
        """Applies normalization transformation."""
        min_val, max_val = self.norm_range
        image = tf.cast(image, tf.float32) / 255.0
        return image * (max_val - min_val) + min_val

    def _validate_norm_range(self):
        """Ensures the normalization range is valid."""
        min_val, max_val = self.norm_range
        if min_val >= max_val:
            raise ValueError(f"❌ Invalid normalization range: {self.norm_range}. min must be < max.")

    def _validate_input(self, image):
        """Ensures the input is a valid TensorFlow image tensor."""
        if not isinstance(image, tf.Tensor):
            raise TypeError(f"❌ Expected TensorFlow tensor, got {type(image)}.")
        if image.shape.ndims != 3:
            raise ValueError(f"❌ Expected 3D image tensor, got shape {image.shape}.")
