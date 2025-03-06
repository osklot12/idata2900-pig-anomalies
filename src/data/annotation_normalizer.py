from typing import Tuple, List, Any

from src.data.annotation_enum_parser import AnnotationEnumParser


class AnnotationNormalizer:
    """Normalizes annotations."""

    @staticmethod
    def normalize_bounding_box(image_dimensions: Tuple[float, float], bounding_box: Tuple[float, float, float, float],
                               new_range: Tuple[float, float]) -> Tuple[float, float, float, float]:
        """
        Normalizes a bounding box to a specified range.

        :param image_dimensions: Tuple containing the image dimensions (width, height) for image associated with bounding box.
        :param bounding_box: Tuple containing the bounding box values (center_x, center_y, width, height).
        :param new_range: Tuple containing the range (min, max) for the normalized bounding box.
        """
        # switch places or range values if the first is greater than the second (e.g (5, 3) -> (3, 5))
        if new_range[0] > new_range[1]:
            new_range = (new_range[1], new_range[0])

        range_interval = new_range[1] - new_range[0]
        normalized_x = (bounding_box[0] / image_dimensions[0]) * range_interval + new_range[0]
        normalized_y = (bounding_box[1] / image_dimensions[1]) * range_interval + new_range[0]
        normalized_w = (bounding_box[2] / image_dimensions[0]) * range_interval
        normalized_h = (bounding_box[3] / image_dimensions[1]) * range_interval

        return normalized_x, normalized_y, normalized_w, normalized_h

    @staticmethod
    def normalize_annotations(
            image_dimensions: Tuple[float, float],
            annotations: List[Tuple[str, float, float, float, float]],
            new_range: Tuple[float, float],
            annotation_parser: AnnotationEnumParser
    ) -> List[Tuple[Any, float, float, float, float]]:
        """
        Normalizes a list of annotations.
        The normalization includes computing a new range for the bounding box and parsing annotation label
        to an enum.

        Args:
            image_dimensions: (width, height) dimensions of the image.
            annotations: List of tuples in the format [(behavior, x, y, w, h), ...].
            new_range: Tuple containing the range (min, max) for the normalized bounding box.
            annotation_parser: An annotation enum parser class.

        Returns:
            List of normalized annotations [(behavior, x, y, w, h), ...].
        """
        if annotation_parser is None:
            raise ValueError("An annotation parser must be provided to map string labels to enums.")

        return [
            (
                annotation_parser.enum_from_str(behavior),
                normalized_x, normalized_y, normalized_w, normalized_h  # Explicit unpacking
            )
            for behavior, x, y, w, h in annotations
            for normalized_x, normalized_y, normalized_w, normalized_h in
            [AnnotationNormalizer.normalize_bounding_box(image_dimensions, (x, y, w, h), new_range)]
        ]
