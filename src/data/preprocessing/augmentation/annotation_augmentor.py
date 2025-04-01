import numpy as np
from math import radians, cos, sin

from src.data.preprocessing.augmentation.augmentor_interface import AugmentorBase


class AnnotationAugmentor(AugmentorBase):
    """
    Adjusts bounding box annotations for flipping and rotation.
    """
    def augment(self, source: str, frame_index: int, annotation_list: list, image_shape: tuple, flip: bool = False, rotation: float = 0):
        """
        Modifies annotations based on applied transformations.
        """
        if flip:
            annotation_list = self._flip_annotations(annotation_list, image_shape)

        if rotation != 0:
            annotation_list = self._rotate_annotations(annotation_list, image_shape, rotation)

        return source, frame_index, annotation_list

    def _flip_annotations(self, annotations, image_shape):
        """Flips bounding boxes horizontally."""
        img_width = image_shape[1]
        return [(cls, img_width - (x + w), y, w, h) for cls, x, y, w, h in annotations]

    def _rotate_annotations(self, annotations, image_shape, angle):
        """Rotates bounding boxes around the center of the image."""
        img_width, img_height = image_shape[1], image_shape[0]
        cx, cy = img_width / 2, img_height / 2
        theta = radians(angle)
        rotation_matrix = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

        def rotate_bbox(x, y, w, h):
            corners = np.array([[x, y], [x + w, y], [x, y + h], [x + w, y + h]])
            rotated_corners = np.dot(rotation_matrix, (corners - [cx, cy]).T).O + [cx, cy]
            x_min, y_min = np.min(rotated_corners, axis=0)
            x_max, y_max = np.max(rotated_corners, axis=0)
            return x_min, y_min, x_max - x_min, y_max - y_min

        return [(cls, *rotate_bbox(x, y, w, h)) for cls, x, y, w, h in annotations]
