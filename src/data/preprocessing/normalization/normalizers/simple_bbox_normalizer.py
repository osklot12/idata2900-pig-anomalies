from typing import Tuple

from src.data.dataclasses.bbox import BBox
from src.data.preprocessing.normalization.normalizers.bbox_normalizer import BBoxNormalizer


class SimpleBBoxNormalizer(BBoxNormalizer):
    """Normalizes bounding box annotations based on image dimensions and target range."""

    def __init__(self, new_range: Tuple[float, float]):
        """
        Initializes the annotation normalizer with fixed image dimensions and a target range.

        Args:
            new_range: tuple (min, max) for the normalized bounding box range
        """
        self.new_range = new_range

        # ensure correct range order
        if self.new_range[0] > self.new_range[1]:
            self.new_range = (self.new_range[1], self.new_range[0])

    def normalize_bounding_box(self, bbox: BBox, resolution: Tuple[int, int]) -> BBox:
        range_interval = self.new_range[1] - self.new_range[0]
        normalized_x = (bbox.x / resolution[0]) * range_interval + self.new_range[0]
        normalized_y = (bbox.y / resolution[1]) * range_interval + self.new_range[0]
        normalized_h = (bbox.height / resolution[1]) * range_interval
        normalized_w = (bbox.width / resolution[0]) * range_interval

        return BBox(normalized_x, normalized_y, normalized_w, normalized_h)