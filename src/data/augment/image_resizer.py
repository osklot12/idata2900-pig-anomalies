import tensorflow as tf
from src.data.augment.augmentor_interface import ProcessorBase

class ImageResizer(ProcessorBase):
    """Handles image resizing to a target dimension."""

    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def process(self, image: tf.Tensor) -> tf.Tensor:
        """Resizes the image to the target dimensions."""
        self._validate_input(image)
        return tf.image.resize(image, self.target_size)

    def _validate_input(self, image):
        """Ensures the input is a valid TensorFlow image tensor."""
        if not isinstance(image, tf.Tensor):
            raise TypeError(f"❌ Expected TensorFlow tensor, got {type(image)}.")
        if image.shape.ndims != 3:
            raise ValueError(f"❌ Expected 3D image tensor, got shape {image.shape}.")
