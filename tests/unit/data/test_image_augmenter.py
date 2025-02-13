import tensorflow as tf
import random
from src.data.pipeline_component import PipelineComponent


class ImageAugmentor(PipelineComponent):
    """
    Image augmentation class that applies various transformations
    such as flipping, rotation (random angles), brightness, contrast, zoom, cropping, and noise.
    """

    def __init__(self, target_size=(224, 224), seed=None):
        super().__init__()
        self.target_size = target_size
        self.seed = seed  # Optional seed for reproducibility
        if self.seed:
            random.seed(self.seed)
            tf.random.set_seed(self.seed)
        self.to_tf_function()

    def process(self, image: tf.Tensor) -> tf.Tensor:
        """Applies diverse augmentations to the image."""
        self._validate_input(image)
        return self._augment(image)

    def _augment(self, image: tf.Tensor) -> tf.Tensor:
        """Applies a series of random augmentations to an image tensor."""

        # Randomly flip the image (left-right / up-down)
        if random.random() > 0.5:
            image = tf.image.flip_left_right(image)
        if random.random() > 0.5:
            image = tf.image.flip_up_down(image)

        # Random rotation (any angle, not just 90-degree multiples)
        image = self._random_rotation(image)

        # Random brightness & contrast adjustment
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)

        # Random zoom (cropping)
        image = self._random_zoom(image)

        # Random translation (shifting)
        image = self._random_translate(image)

        # Random noise
        image = self._add_random_noise(image)

        return image

    def _random_rotation(self, image: tf.Tensor) -> tf.Tensor:
        """Rotates image by a small random angle (-30 to 30 degrees)."""
        angle = random.uniform(-30, 30)  # Random angle in degrees
        radians = angle * (3.1415926535 / 180)  # Convert to radians
        return tfa.image.rotate(image, radians)  # Requires TensorFlow Addons

    def _random_zoom(self, image: tf.Tensor) -> tf.Tensor:
        """Applies random zoom by cropping and resizing."""
        crop_fraction = random.uniform(0.7, 1.0)  # Keep 70-100% of the original image
        crop_size = [int(self.target_size[0] * crop_fraction), int(self.target_size[1] * crop_fraction), 3]
        image = tf.image.resize_with_crop_or_pad(image, crop_size[0], crop_size[1])
        return tf.image.resize(image, self.target_size)

    def _random_translate(self, image: tf.Tensor) -> tf.Tensor:
        """Applies random translation (shifting)."""
        translations = [random.randint(-20, 20), random.randint(-20, 20)]  # Shift by up to 20 pixels
        return tfa.image.translate(image, translations)

    def _add_random_noise(self, image: tf.Tensor) -> tf.Tensor:
        """Adds random Gaussian noise to the image."""
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05, dtype=tf.float32)
        return tf.clip_by_value(image + noise, 0.0, 1.0)  # Keep within valid pixel range

    def _validate_input(self, image):
        """Ensures the input is a valid TensorFlow image tensor."""
        if not isinstance(image, tf.Tensor):
            raise TypeError(f"❌ Expected TensorFlow tensor, but got {type(image)} instead.")

        if image.shape.ndims is None or image.shape.ndims != 3:
            raise ValueError(f"❌ Expected 3D image tensor [H, W, C], but got shape {image.shape}.")
