from typing import Tuple, List, Any, Type

from src.data.annotation_enum_parser import AnnotationEnumParser


class BBoxNormalizer:
    """Normalizes bounding box annotations based on image dimensions and target range."""

    def __init__(self, image_dimensions: Tuple[float, float], new_range: Tuple[float, float],
                 annotation_parser: Type[AnnotationEnumParser]):
        """
        Initializes the annotation normalizer with fixed image dimensions and a target range.

        Args:
            image_dimensions: Tuple (width, height) representing the original image dimensions.
            new_range: Tuple (min, max) for the normalized bounding box range.
            annotation_parser: Enum parser for converting annotation labels.
        """
        self.image_width, self.image_height = image_dimensions
        self.new_range = new_range
        self.annotation_parser = annotation_parser

        # ensure correct range order
        if self.new_range[0] > self.new_range[1]:
            self.new_range = (self.new_range[1], self.new_range[0])

    def set_image_dimensions(self, image_dimensions: Tuple[float, float]):
        """
        Sets the image dimensions.

        Args:
            image_dimensions: Tuple (width, height) representing the original image dimensions.
        """
        self.image_width, self.image_height = image_dimensions

    def normalize_bounding_box(self, bounding_box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        """
        Normalizes a bounding box to the configured range.

        Args:
            bounding_box: (center_x, center_y, width, height)

        Returns:
            Normalized (x, y, w, h) within `self.new_range`
        """
        range_interval = self.new_range[1] - self.new_range[0]
        normalized_x = (bounding_box[0] / self.image_width) * range_interval + self.new_range[0]
        normalized_y = (bounding_box[1] / self.image_height) * range_interval + self.new_range[0]
        normalized_w = (bounding_box[2] / self.image_width) * range_interval
        normalized_h = (bounding_box[3] / self.image_height) * range_interval

        return normalized_x, normalized_y, normalized_w, normalized_h

    def normalize_annotations(self, annotations: List[Tuple[str, float, float, float, float]]) -> List[Tuple[Any, float, float, float, float]]:
        """
        Normalizes a list of annotations using configured dimensions and range.

        Args:
            annotations: List of tuples [(behavior, x, y, w, h), ...]

        Returns:
            List of normalized annotations [(behavior_enum, x, y, w, h), ...]
        """
        if self.annotation_parser is None:
            raise ValueError("An annotation parser must be provided to map string labels to enums.")

        normalized_annotations: List[Tuple[Any, float, float, float, float]] = []

        for behavior, x, y, w, h in annotations:
            normalized_bbox = self.normalize_bounding_box((x, y, w, h))  # Ensures correct return type

            normalized_annotations.append((
                self.annotation_parser.enum_from_str(behavior),
                float(normalized_bbox[0]),  # Explicit float conversion
                float(normalized_bbox[1]),
                float(normalized_bbox[2]),
                float(normalized_bbox[3]),
            ))

        return normalized_annotations

