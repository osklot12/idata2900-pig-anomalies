import tensorflow as tf
import random
from src.data.pipeline_component import PipelineComponent


class ImageAugmentor(PipelineComponent):
    """
    Image augmentation class that applies diverse random transformations,
    ensuring each augmented image is different.
    """

    def __init__(self, target_size=(224, 224), seed=None):
        super().__init__()
        self.target_size = target_size
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
            tf.random.set_seed(self.seed)
        self.to_tf_function()

    def process(self, image: tf.Tensor) -> tf.Tensor:
        """Applies a variety of augmentations to ensure each image is unique."""
        self._validate_input(image)
        return self._augment(tf.identity(image))  # Clone image to prevent consistency issues

    def _augment(self, image: tf.Tensor) -> tf.Tensor:
        """Applies randomized transformations to create diverse variations."""

        # Use tf.cond() to ensure independent randomness for flipping
        image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: tf.image.flip_left_right(image), lambda: image)
        image = tf.cond(tf.random.uniform([], 0, 1) > 0.5, lambda: tf.image.flip_up_down(image), lambda: image)

        # Random Rotation (0, 90, 180, or 270 degrees)
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)  # Generates 0, 1, 2, or 3
        image = tf.image.rot90(image, k=k)

        # Random Brightness & Contrast Adjustments (which were already working)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)

        # Random Zoom (cropping + resizing back), applied individually
        crop_fraction = tf.random.uniform([], 0.6, 1.0)  # Random crop between 60%-100%
        crop_size = [tf.cast(self.target_size[0] * crop_fraction, tf.int32),
                     tf.cast(self.target_size[1] * crop_fraction, tf.int32), 3]

        image = tf.image.resize_with_crop_or_pad(image, crop_size[0], crop_size[1])
        image = tf.image.resize(image, self.target_size)

        return image

    def _validate_input(self, image):
        """Ensures the input is a valid TensorFlow image tensor."""
        if not isinstance(image, tf.Tensor):
            raise TypeError(f"❌ Expected TensorFlow tensor, but got {type(image)} instead.")

        if image.shape.ndims is None or image.shape.ndims != 3:
            raise ValueError(f"❌ Expected 3D image tensor [H, W, C], but got shape {image.shape}.")
