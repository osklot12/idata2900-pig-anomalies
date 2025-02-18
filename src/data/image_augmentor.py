import tensorflow as tf
import random
import numpy as np
from math import radians, cos, sin
from src.data.pipeline_component import PipelineComponent


class ImageAugmentor(PipelineComponent):
    """
    Image augmentation class that applies transformations such as horizontal mirroring, rotation,
    brightness, contrast, and ensures bounding box updates.
    """

    def __init__(self, target_size=(224, 224), seed=None):
        super().__init__()
        self.target_size = target_size
        self.seed = seed
        if self.seed:
            random.seed(self.seed)
            tf.random.set_seed(self.seed)
        self.to_tf_function()

    def process(self, image: tf.Tensor, annotation: dict, rotation_angle=None) -> (tf.Tensor, dict):
        """Applies a variety of augmentations and returns the transformed image and bounding box."""
        self._validate_input(image)

        # Create a copy of the annotation to avoid modifying the original
        annotation_copy = annotation.copy()
        bbox = annotation_copy.get("bounding_box", None)

        return self._augment(image, bbox, annotation_copy, rotation_angle)

    def _augment(self, image: tf.Tensor, bbox, annotation, rotation_angle=None) -> (tf.Tensor, dict):
        """Applies randomized transformations to create diverse variations."""
        orig_height, orig_width, _ = image.shape

        # Horizontal mirroring only
        flip_horizontal = random.choice([True, False])

        if flip_horizontal:
            image = tf.image.flip_left_right(image)

        # Random rotation (small angles, e.g., -15° to 15°)
        if rotation_angle is None:
            rotation_angle = random.uniform(-15, 15)  # Restricted to ±15°
        image = self._apply_rotation(image, rotation_angle)

        # Random brightness & contrast
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.7, upper=1.3)

        # Adjust bounding box based on augmentations
        if bbox is not None:
            updated_bbox = self._adjust_bbox(bbox, orig_width, orig_height, flip_horizontal, rotation_angle)
            annotation["bounding_box"] = updated_bbox  # Modify only the copied annotation

        return image, annotation

    def _apply_rotation(self, image: tf.Tensor, angle: float) -> tf.Tensor:
        """Rotates an image by a given angle in degrees using TensorFlow's affine transformation."""
        radians_angle = radians(angle)

        # Compute transformation matrix
        transform_matrix = tf.constant([
            [cos(radians_angle), -sin(radians_angle), 0,
             sin(radians_angle), cos(radians_angle), 0,
             0, 0]
        ], dtype=tf.float32)

        # Reshape to match TensorFlow's requirement (1 x 8)
        transform_matrix = tf.reshape(transform_matrix, [1, 8])

        # Apply transformation
        rotated_image = tf.raw_ops.ImageProjectiveTransformV3(
            images=tf.expand_dims(image, axis=0),
            transforms=transform_matrix,
            interpolation="BILINEAR",
            fill_mode="CONSTANT",
            output_shape=tf.shape(image)[:2],
            fill_value=0  # Black background fill
        )

        return tf.squeeze(rotated_image, axis=0)  # Remove batch dimension

    def _adjust_bbox(self, bbox, orig_width, orig_height, flip_horizontal, rotation_angle):
        """Adjusts bounding box coordinates after transformations while keeping it axis-aligned."""
        if bbox is None:
            return None

        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        cx, cy = orig_width / 2, orig_height / 2  # Center of rotation
        theta = radians(rotation_angle)

        # Define four bounding box corner points
        corners = np.array([
            [x, y],  # Top-left
            [x + w, y],  # Top-right
            [x, y + h],  # Bottom-left
            [x + w, y + h]  # Bottom-right
        ])

        # Apply rotation to each corner
        rotated_corners = []
        for px, py in corners:
            x_new = cos(theta) * (px - cx) - sin(theta) * (py - cy) + cx
            y_new = sin(theta) * (px - cx) + cos(theta) * (py - cy) + cy
            rotated_corners.append([x_new, y_new])

        rotated_corners = np.array(rotated_corners)

        # Compute the new bounding box (smallest enclosing AABB)
        x_min, y_min = np.min(rotated_corners, axis=0)
        x_max, y_max = np.max(rotated_corners, axis=0)

        # Handle horizontal flipping
        if flip_horizontal:
            x_min = orig_width - (x_min + (x_max - x_min))

        return {"x": float(x_min), "y": float(y_min), "w": float(x_max - x_min), "h": float(y_max - y_min)}

    def _validate_input(self, image):
        """Ensures the input is a valid TensorFlow image tensor."""
        if not isinstance(image, tf.Tensor):
            raise TypeError(f"❌ Expected TensorFlow tensor, but got {type(image)} instead.")
        if image.shape.ndims is None or image.shape.ndims != 3:
            raise ValueError(f"❌ Expected 3D image tensor [H, W, C], but got shape {image.shape}.")
