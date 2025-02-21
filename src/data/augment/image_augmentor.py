import tensorflow as tf
import random
from math import radians, cos, sin
from src.data.augment.augmentor_interface import AugmentorBase

class ImageAugmentor(AugmentorBase):
    """Applies image augmentations like flipping, rotation, brightness, and contrast."""

    def __init__(self, seed=None):
        if seed:
            random.seed(seed)
            tf.random.set_seed(seed)

    def augment(self, image: tf.Tensor, annotation=None, rotation: float = 0, flip: bool = False) -> tf.Tensor:
        """Applies augmentations with specified rotation and flip."""
        if flip:
            image = tf.image.flip_left_right(image)
        if rotation != 0:
            image = self._rotate(image, rotation)

        image = self._random_brightness_contrast(image)
        return image

    def _random_brightness_contrast(self, image):
        """Applies random brightness and contrast adjustments."""
        image = tf.image.random_brightness(image, max_delta=0.2)
        return tf.image.random_contrast(image, lower=0.7, upper=1.3)

    def _rotate(self, image: tf.Tensor, angle: float) -> tf.Tensor:
        """Rotates an image by a given angle using TensorFlow affine transformation."""
        radians_angle = radians(angle)
        transform_matrix = tf.reshape(tf.constant([
            cos(radians_angle), -sin(radians_angle), 0,
            sin(radians_angle), cos(radians_angle), 0,
            0, 0
        ], dtype=tf.float32), [1, 8])

        rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, axis=0),
            transforms=transform_matrix,
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            output_shape=tf.shape(image)[:2],
            fill_value=0
        )

        return tf.squeeze(rotated_image, axis=0)
