from typing import List, Tuple
import random

import cv2
import numpy as np

from src.data.dataclasses.annotated_bbox import AnnotatedBBox
from src.data.dataclasses.bbox import BBox
from src.data.dataclasses.streamed_annotated_frame import StreamedAnnotatedFrame
from src.data.preprocessing.preprocessor import Preprocessor


class AnnotatedFrameAugmentor(Preprocessor[StreamedAnnotatedFrame]):
    """Augments annotated frames."""

    @staticmethod
    def _compute_shift_and_transform_matrix(t: np.ndarray, cx, cy) -> np.ndarray:
        """Computes a transformation matrix that shifts to the origin before transforming."""
        return (
                AnnotatedFrameAugmentor._compute_translation_matrix(cx, cy) @
                t @
                AnnotatedFrameAugmentor._compute_translation_matrix(-cx, -cy)
        )

    @staticmethod
    def _compute_translation_matrix(dx: float, dy: float) -> np.ndarray:
        """Computes the translation matrix for a shift by (dx, dy)."""
        return np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def _compute_dilation_matrix(factor: float) -> np.ndarray:
        """Computes the dilation matrix from given dilation factor."""
        return np.array([
            [factor, 0, 0],
            [0, factor, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def _compute_rotation_matrix(degrees: float) -> np.ndarray:
        """Computes the rotation matrix from given degrees."""
        radians = np.deg2rad(degrees)
        cos_theta = np.cos(radians)
        sin_theta = np.sin(radians)

        return np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ], dtype=np.float32)

    @staticmethod
    def _compute_rotation_degrees(max_degrees: float) -> float:
        """Computes a random rotation degree."""
        return random.uniform(-max_degrees, max_degrees)

    @staticmethod
    def _compute_dilation_factor(max_factor: float) -> float:
        """Computes a random dilation factor."""
        return random.uniform(1.0, max_factor)

    @staticmethod
    def _compute_max_dilation_factor(annotations: List[AnnotatedBBox]) -> float:
        """Computes the dilation faction."""
        closest_edge = 1.0

        for annotation in annotations:
            dist = AnnotatedFrameAugmentor._compute_distance_closest_edge(annotation.bbox)
            if dist < closest_edge:
                closest_edge = dist

        epsilon = 1e-6
        return 0.5 / max(closest_edge, epsilon)

    @staticmethod
    def _get_corner_coords(bbox: BBox) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Converts the bounding box into two point coordinates: top left and bottom right."""
        return (bbox.x, bbox.y), (bbox.x + bbox.width, bbox.y + bbox.height)

    @staticmethod
    def _compute_distance_closest_edge(bbox: BBox) -> float:
        """Computes the distance to the edge closest to the bounding box."""
        (x_min, y_min), (x_max, y_max) = AnnotatedFrameAugmentor._get_corner_coords(bbox)

        distances = [
            x_min, y_min, 1 - x_max, 1 - y_max
        ]

        return min(distances)

    @staticmethod
    def _get_corner_points(bbox: BBox) -> List[np.ndarray]:
        """Extracts the corner point coordinates from the bounding box."""
        return [
            np.array([bbox.x, bbox.y, 1]),
            np.array([bbox.x + bbox.width, bbox.y, 1]),
            np.array([bbox.x, bbox.y + bbox.height, 1]),
            np.array([bbox.x + bbox.width, bbox.y + bbox.height, 1])
        ]

    @staticmethod
    def _get_absolute_bbox(bbox: BBox, frame_shape: Tuple[float, float]) -> BBox:
        """Converts a normalized bounding box to absolute pixel values."""
        width, height = frame_shape
        return BBox(
            x=bbox.x * width,
            y=bbox.y * height,
            width=bbox.width * width,
            height=bbox.height * height
        )

    @staticmethod
    def _get_normalized_bbox(bbox: BBox, frame_shape: Tuple[float, float]) -> BBox:
        """Converts an absolute bounding box to normalized values."""
        width, height = frame_shape
        return BBox(
            x=bbox.x / width,
            y=bbox.y / height,
            width=bbox.width / width,
            height=bbox.height / height
        )

    @staticmethod
    def _augment_annotations(annotations: List[AnnotatedBBox], transformation: np.ndarray,
                             frame_shape: Tuple[int, int]) -> List[AnnotatedBBox]:
        """Augments annotations using given transformation."""
        augmented = []

        for anno in annotations:
            bbox = anno.bbox
            absolute_bbox = AnnotatedFrameAugmentor._get_absolute_bbox(bbox, frame_shape)
            corners = AnnotatedFrameAugmentor._get_corner_points(absolute_bbox)

            transformed = [transformation @ corner for corner in corners]
            xs = [pt[0] for pt in transformed]
            ys = [pt[1] for pt in transformed]

            x_min, y_min = min(xs), min(ys)
            x_max, y_max = max(xs), max(ys)

            width = x_max - x_min
            height = y_max - y_min

            augmented.append(AnnotatedBBox(
                cls=anno.cls,
                bbox=AnnotatedFrameAugmentor._get_normalized_bbox(
                    BBox(x=x_min, y=y_min, width=width, height=height),
                    frame_shape
                )
            ))

        return augmented

    @staticmethod
    def _augment_frame(frame: np.ndarray, transformation: np.ndarray) -> np.ndarray:
        """Applies an affine transformation to the image (frame)."""
        height, width = frame.shape[:2]

        affine_2x3 = transformation[:2, :]

        return cv2.warpAffine(
            frame,
            affine_2x3,
            dsize=(width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

    def process(self, instance: StreamedAnnotatedFrame) -> List[StreamedAnnotatedFrame]:
        rot_deg = self._compute_rotation_degrees(10)
        rot_matrix = self._compute_rotation_matrix(rot_deg)

        dilate_factor = self._compute_dilation_factor(1.1)
        dilate_matrix = self._compute_dilation_matrix(dilate_factor)

        rot_dia_matrix = dilate_matrix @ rot_matrix

        frame = instance.frame
        frame_width, frame_height = (frame.shape[1], frame.shape[0])
        frame_center = (frame_width / 2, frame_height / 2)
        affine_matrix = self._compute_shift_and_transform_matrix(rot_dia_matrix, *frame_center)

        aug_frame = self._augment_frame(frame, affine_matrix)
        aug_anno = self._augment_annotations(instance.annotations, affine_matrix, (frame_width, frame_height))

        return [
            StreamedAnnotatedFrame(
                source=instance.source,
                index=instance.index,
                frame=aug_frame,
                annotations=aug_anno
            )
        ]
