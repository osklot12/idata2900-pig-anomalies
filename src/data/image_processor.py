import tensorflow as tf
from src.data.pipeline_component import PipelineComponent

class ImageProcessor(PipelineComponent):
    """
    Image processing class that resizes and normalizes images.
    Ensures proper error handling for invalid inputs.
    """

    def __init__(self, target_size=(224, 224), norm_range=(0, 1)):
        super().__init__()
        self.target_size = target_size
        self.norm_range = norm_range
        self._validate_norm_range()
        self.to_tf_function()

    def _validate_norm_range(self):
        """Ensures the normalization range is valid."""
        min_val, max_val = self.norm_range
        if min_val >= max_val:
            raise ValueError(f"❌ Invalid normalization range: {self.norm_range}. min must be < max.")

    def process(self, image: tf.Tensor) -> tf.Tensor:
        """Processes an image tensor by resizing and normalizing it."""
        self._validate_input(image)  # Validate before processing

        image = self._resize(image)
        image = self._normalize(image)
        return image

    def _validate_input(self, image):
        """Validates the input tensor."""
        if not isinstance(image, tf.Tensor):
            raise TypeError(f"❌ Expected TensorFlow tensor, but got {type(image)} instead.")

        if image.shape.ndims is None:
            raise ValueError("❌ Received an unknown shape tensor. Ensure valid input.")

        if image.shape.ndims != 3:
            raise ValueError(f"❌ Expected 3D image tensor [H, W, C], but got shape {image.shape}.")

        # Prevent graph execution errors using tf.cond()
        is_empty = tf.equal(tf.size(image), 0)
        tf.debugging.assert_equal(is_empty, False, message="❌ Input tensor is empty! Ensure valid image data.")

    def _resize(self, image: tf.Tensor) -> tf.Tensor:
        """Resizes the image to the target dimensions."""
        return tf.image.resize(image, self.target_size)

    def _normalize(self, image: tf.Tensor) -> tf.Tensor:
        """Normalizes the image to the specified range (min, max)."""
        min_val, max_val = self.norm_range
        if min_val >= max_val:
            raise ValueError(f"❌ Invalid normalization range: {self.norm_range}. min must be < max.")

        image = tf.cast(image, tf.float32) / 255.0  # Scale to [0,1]
        image = image * (max_val - min_val) + min_val  # Scale to [min, max]
        return image

    def configure(self, target_size=None, norm_range=None):
        """Allows dynamic configuration of processing settings."""
        if target_size:
            self.target_size = target_size
        if norm_range:
            self.norm_range = norm_range
            self._validate_norm_range()

