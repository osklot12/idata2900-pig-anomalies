import numpy as np
from math import radians, cos, sin
from src.data.augment.augmentor_interface import AugmentorBase


class JsonAugmentor(AugmentorBase):
    """Applies transformations to bounding box annotations."""

    def augment(self, image, annotation: dict, rotation: float = 0, flip: bool = False) -> dict:
        """Modifies annotations based on applied transformations."""
        if "bounding_box" not in annotation:
            return annotation

        img_shape = image.shape if image is not None else (224, 224, 3)  # Default shape

        annotation["bounding_box"] = self._adjust_bbox(
            annotation["bounding_box"], img_shape, flip, rotation
        )
        return annotation

    def _adjust_bbox(self, bbox, image_shape, flip_horizontal, rotation_angle):
        """Adjusts bounding box after transformations."""
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        img_width, img_height = image_shape[1], image_shape[0]
        cx, cy = img_width / 2, img_height / 2

        corners = np.array([
            [x, y], [x + w, y], [x, y + h], [x + w, y + h]
        ])

        if rotation_angle != 0:
            corners = self._rotate_corners(corners, cx, cy, rotation_angle)

        x_min, y_min = np.min(corners, axis=0)
        x_max, y_max = np.max(corners, axis=0)

        if flip_horizontal:
            x_min = img_width - (x_min + (x_max - x_min))

        return {"x": float(x_min), "y": float(y_min), "w": float(x_max - x_min), "h": float(y_max - y_min)}

    def _rotate_corners(self, corners, cx, cy, angle):
        """Rotates bounding box corners around image center."""
        theta = radians(angle)
        rotation_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
        return np.dot(corners - [cx, cy], rotation_matrix.T) + [cx, cy]
