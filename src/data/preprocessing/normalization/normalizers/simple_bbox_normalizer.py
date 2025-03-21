from typing import Tuple

from src.data.dataclasses.bounding_box import BoundingBox
from src.data.preprocessing.normalization.normalizers.bbox_normalization_strategy import BBoxNormalizationStrategy


class SimpleBBoxNormalizer(BBoxNormalizationStrategy):
    """Normalizes bounding box annotations based on image dimensions and target range."""

    def __init__(self, image_dimensions: Tuple[float, float], new_range: Tuple[float, float]):
        """
        Initializes the annotation normalizer with fixed image dimensions and a target range.

        Args:
            image_dimensions: Tuple (width, height) representing the original image dimensions.
            new_range: Tuple (min, max) for the normalized bounding box range.
        """
        self.image_width, self.image_height = image_dimensions
        self.new_range = new_range

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

    def normalize_bounding_box(self, bbox: BoundingBox) -> BoundingBox:
        """
        Normalizes a bounding box to the configured range.

        Args:
            bbox (BoundingBox): the bounding box to normalize

        Returns:
            BoundingBox: the normalized bounding box
        """
        range_interval = self.new_range[1] - self.new_range[0]
        normalized_x = (bbox.center_x / self.image_width) * range_interval + self.new_range[0]
        normalized_y = (bbox.center_y / self.image_height) * range_interval + self.new_range[0]
        normalized_h = (bbox.height / self.image_height) * range_interval
        normalized_w = (bbox.width / self.image_width) * range_interval

        return BoundingBox(normalized_x, normalized_y, normalized_w, normalized_h)